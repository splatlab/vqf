/*
 * ============================================================================
 *
 *       Filename:  ququ_filter.c
 *
 *         Author:  Prashant Pandey (), ppandey2@cs.cmu.edu
 *   Organization:  Carnegie Mellon University
 *
 * ============================================================================
 */

#include <iostream>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>  // portable to all x86 compilers
#include <tmmintrin.h>

#include "ququ_filter.h"

#define MAX_VALUE(nbits) ((1ULL << (nbits)) - 1)
#define BITMASK(nbits)                                    \
  ((nbits) == 64 ? 0xffffffffffffffff : MAX_VALUE(nbits))

#define SEED 2038074761
#define QUQU_SLOTS_PER_BLOCK 51

static inline int word_rank(uint64_t val) {
	asm("popcnt %[val], %[val]"
			: [val] "+r" (val)
			:
			: "cc");
	return val;
}

// Returns the position of the rank'th 1.  (rank = 0 returns the 1st 1)
// Returns 64 if there are fewer than rank+1 1s.
static inline uint64_t word_select(uint64_t val, int rank) {
	uint64_t i = 1ULL << rank;
	asm("pdep %[val], %[mask], %[val]"
			: [val] "+r" (val)
			: [mask] "r" (i));
	asm("tzcnt %[bit], %[index]"
			: [index] "=r" (i)
			: [bit] "g" (val)
			: "cc");
	return i;
}

// select(vec, 0) -> -1
// select(vec, i) -> 128, if i > popcnt(vec)
int64_t select_128(__uint128_t vector, uint64_t rank) {
	uint64_t lower_word = vector & BITMASK(64);
	uint64_t lower_rank = word_rank(lower_word);
	if (lower_rank > rank) {
		return word_select(lower_word, rank);
	} else {
		rank = rank - lower_rank;
		uint64_t higher_word = vector >> 64;
		if ((uint64_t)word_rank(higher_word) > rank) {
			return word_select(higher_word, rank) + 64;
		} else {
			return 128;
		}
	}
}

#define SHUFFLE_SIZE 32

//assumes little endian
void print_bits(__uint128_t num, int numbits)
{
  int i;
  for (i = 0 ; i < numbits; i++)
    printf("%d", ((num >> i) & 1) == 1);
	puts("");
}

void print_tags(uint8_t *tags, uint32_t size) {
	for (uint8_t i = 0; i < size; i++)
		printf("%d ", (uint32_t)tags[i]);
	printf("\n");
}

void print_block(ququ_filter *filter, uint64_t block_index) {
	printf("block index: %ld\n", block_index);
	printf("metadata: ");
	__uint128_t md = filter->blocks[block_index].md;
	print_bits(md, 102);
	printf("tags: ");
	print_tags(reinterpret_cast<uint8_t *>(&filter->blocks[block_index].tags), 51);
}

const __m256i K0 = _mm256_setr_epi8(
    0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70,
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0);

const __m256i K1 = _mm256_setr_epi8(
    0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
    0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70);

inline __m256i cross_lane_shuffle(const __m256i & value, const __m256i & shuffle)
{
    return _mm256_or_si256(_mm256_shuffle_epi8(value, _mm256_add_epi8(shuffle, K0)),
        _mm256_shuffle_epi8(_mm256_permute4x64_epi64(value, 0x4E), _mm256_add_epi8(shuffle, K1)));
}

void shuffle_256(uint8_t *source, uint8_t *map) {
	__m256i vector = _mm256_loadu_si256(reinterpret_cast<__m256i*>(source));
	__m256i shuffle = _mm256_loadu_si256(reinterpret_cast<__m256i*>(map));

	vector = cross_lane_shuffle(vector, shuffle);
	_mm256_storeu_si256(reinterpret_cast<__m256i*>(source), vector);
}

void update_tags(ququ_block *block, uint8_t index, uint8_t tag) {
  memmove(&block->tags[index + 1], &block->tags[index], 50 - index);
  block->tags[index] = tag;
}


__uint128_t update_md(__uint128_t md, uint8_t index, uint8_t bit) {
  static const __uint128_t one = 1;
  
  __uint128_t unshifted_mask = (one << index) - 1;
  __uint128_t unshifted_md = md & unshifted_mask;
  __uint128_t shifted_md = (md & ~unshifted_mask) << 1;
  __uint128_t bit128 = (((__uint128_t)bit) << index);
  return shifted_md | bit128 | unshifted_md;
}

// number of 0s in the metadata is the number of tags.
uint64_t get_block_load(__uint128_t vector) {
	uint64_t lower_word = vector & BITMASK(64);
	uint64_t higher_word = vector >> 64;
	uint64_t popcnt = word_rank(lower_word) + word_rank(higher_word);
	return (102 - popcnt);
}

// Create n/log(n) blocks of log(n) slots.
// log(n) is 51 given a cache line size.
// n/51 blocks.
int ququ_init(ququ_filter *filter, uint64_t nslots) {
	assert(word_rank(nslots) == 1); /* nslots must be a power of 2 */

	filter->metadata = (ququ_metadata*)malloc(sizeof(ququ_metadata));
	if (filter->metadata == NULL) {
		perror("Can't allocate memory for metadata");
		exit(EXIT_FAILURE);
	}
	uint64_t total_blocks = nslots/QUQU_SLOTS_PER_BLOCK;
	uint64_t total_slots = nslots;
	uint64_t total_q_bits = 0;
	while (total_slots > 1) {
		total_slots >>= 1;
		total_q_bits++;
	}

	filter->metadata->total_size_in_bytes = sizeof(ququ_block) * total_blocks;
	filter->metadata->seed = SEED;
	filter->metadata->nslots = nslots;
	filter->metadata->key_bits = total_q_bits + 8;
	filter->metadata->key_remainder_bits = 8;
	filter->metadata->range = MAX_VALUE(filter->metadata->key_bits);
	filter->metadata->nblocks = total_blocks;
	filter->metadata->nelts = 0;

	filter->blocks = (ququ_block*)malloc(filter->metadata->total_size_in_bytes);
	if (filter->metadata == NULL) {
		perror("Can't allocate memory for blocks");
		exit(EXIT_FAILURE);
	}
	// memset to 1
	for (uint64_t i = 0; i < total_blocks; i++) {
		__uint128_t set_vector = UINT64_MAX;
		set_vector = set_vector << 64 | UINT64_MAX;
		filter->blocks[i].md = set_vector;
	}

	return 0;
}

// If the item goes in the i'th slot (starting from 0) in the block then
// find the i'th 0 in the metadata, insert a 1 after that and shift the rest
// by 1 bit.
// Insert the new tag at the end of its run and shift the rest by 1 slot.
int ququ_insert(ququ_filter *filter, __uint128_t hash) {
	uint64_t tag = hash & BITMASK(filter->metadata->key_remainder_bits);
	uint64_t block_index = hash >> filter->metadata->key_remainder_bits;
	uint64_t alt_block_index = ((block_index ^ (tag * 0x5bd1e995)) %
															filter->metadata->range) >>
		filter->metadata->key_remainder_bits;

	uint64_t primary_load =	get_block_load(filter->blocks[block_index /
																				 QUQU_SLOTS_PER_BLOCK].md); 
	uint64_t alt_load =	get_block_load(filter->blocks[alt_block_index /
																		 QUQU_SLOTS_PER_BLOCK].md);
	if (primary_load == QUQU_SLOTS_PER_BLOCK && alt_load ==
			QUQU_SLOTS_PER_BLOCK) {
		fprintf(stderr, "ququ filter is full.");
		exit(EXIT_FAILURE);
	}
	// pick the least loaded block
	if (alt_load < primary_load) {
		block_index = alt_block_index;
	}

	uint64_t index = block_index / QUQU_SLOTS_PER_BLOCK;
	uint64_t offset = block_index % QUQU_SLOTS_PER_BLOCK;

	uint64_t select_index = select_128(filter->blocks[index].md, offset);
	uint64_t slot_index = select_index - offset;
	
	printf("tag: %ld offset: %ld\n", tag, offset);
	print_block(filter, index);

	update_tags(&filter->blocks[index], slot_index,	tag);
	filter->blocks[index].md = update_md(filter->blocks[index].md, select_index, 0);
	print_block(filter, index);
	return 0;
}

bool check_tags(ququ_filter *filter, uint8_t tag, uint64_t block_index) {
	uint64_t index = block_index / QUQU_SLOTS_PER_BLOCK;
	uint64_t offset = block_index % QUQU_SLOTS_PER_BLOCK;

	uint64_t start, end;
	if (offset == 0) {
		start = 0;
	} else {
		start = select_128(filter->blocks[index].md, offset - 1) - offset + 1;
	}
	end = select_128(filter->blocks[index].md, offset) - offset;

	for (uint64_t i = start; i < end; i++) {
		if (tag == filter->blocks[index].tags[i])
			return true;
	}
	return false;
}

// If the item goes in the i'th slot (starting from 0) in the block then
// select(i) - i is the slot index for the end of the run.
bool ququ_is_present(ququ_filter *filter, __uint128_t hash) {
	uint64_t tag = hash & BITMASK(filter->metadata->key_remainder_bits);
	uint64_t block_index = hash >> filter->metadata->key_remainder_bits;
	uint64_t alt_block_index = ((block_index ^ (tag * 0x5bd1e995)) %
															filter->metadata->range) >>
		filter->metadata->key_remainder_bits;

	bool ret = check_tags(filter, tag, block_index) ? true : check_tags(filter, tag,
																																	alt_block_index);
	if (!ret) {
		printf("tag: %ld offset: %ld\n", tag, block_index % QUQU_SLOTS_PER_BLOCK);
		print_block(filter, block_index / QUQU_SLOTS_PER_BLOCK);
		print_block(filter, alt_block_index / QUQU_SLOTS_PER_BLOCK);
	}
	return ret;
}

