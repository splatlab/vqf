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

#define QUQU_SLOTS_PER_BLOCK 48
#define QUQU_BUCKETS_PER_BLOCK 80

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
static inline int64_t select_128(__uint128_t vector, uint64_t rank) {
	uint64_t lower_word = vector & BITMASK(64);
	uint64_t lower_rank = word_rank(lower_word);
	if (lower_rank > rank) {
		return word_select(lower_word, rank);
	} else {
		rank = rank - lower_rank;
		uint64_t higher_word = vector >> 64;
    return word_select(higher_word, rank) + 64;
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
	print_bits(md, QUQU_BUCKETS_PER_BLOCK + QUQU_SLOTS_PER_BLOCK);
	printf("tags: ");
	print_tags(filter->blocks[block_index].tags, QUQU_SLOTS_PER_BLOCK);
}

#if 1
static inline void update_tags(ququ_block * restrict block, uint8_t index, uint8_t tag) {
	memmove(&block->tags[index + 1], &block->tags[index], sizeof(block->tags) / sizeof(block->tags[0]) - index - 1);
	block->tags[index] = tag;
}

#else
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

void shuffle_256(uint8_t * restrict source, uint8_t * restrict map) { 
  __m256i vector = _mm256_loadu_si256(reinterpret_cast<__m256i*>(source)); 
  __m256i shuffle = _mm256_loadu_si256(reinterpret_cast<__m256i*>(map)); 

  vector = cross_lane_shuffle(vector, shuffle); 
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(source), vector); 
} 

static inline void update_tags(uint8_t * restrict block, uint8_t index, uint8_t tag) {
	index = index + 13;		// offset index based on the md size
	if (index < SHUFFLE_SIZE) {	// change in the first 32-bytes. Must move both halves.
		/*Move first 32-bytes*/
		// create a mapping vector
		uint8_t map[SHUFFLE_SIZE];
		for (uint8_t i = 0, j = 0; i < SHUFFLE_SIZE; i++) {
			if (i == index) {
				map[i] = SHUFFLE_SIZE - 1;
			} else {
				map[i] = j++;
			}
		}
		uint8_t source[SHUFFLE_SIZE];
		memcpy(source, block, SHUFFLE_SIZE);
		uint8_t overflow_tag = source[SHUFFLE_SIZE - 1];
		// add the new tag as the last index
		source[SHUFFLE_SIZE - 1] = tag;
		/*print_tags(source, SHUFFLE_SIZE);*/
		shuffle_256(source, map);
		/*print_tags(source, SHUFFLE_SIZE);*/
		memcpy(block, source, SHUFFLE_SIZE);

		/*move second 32-bytes*/
		for (uint8_t i = 0, j = 0; i < SHUFFLE_SIZE; i++) {
			map[i] = j++;
		}
		memcpy(source, block + SHUFFLE_SIZE, SHUFFLE_SIZE);
		shuffle_256(source, map);
		source[0] = overflow_tag;
		memcpy(block + SHUFFLE_SIZE, source, SHUFFLE_SIZE);
	} else {	// change in the second 32-bytes chunk. Only affects the second half.
		index = index - SHUFFLE_SIZE;
		// create a mapping vector
		uint8_t map[SHUFFLE_SIZE];
		for (uint8_t i = 0, j = 0; i < SHUFFLE_SIZE; i++) {
			if (i == index) {
				map[i] = SHUFFLE_SIZE - 1;
			} else {
				map[i] = j++;
			}
		}
		uint8_t source[SHUFFLE_SIZE];
		memcpy(source, block + SHUFFLE_SIZE, SHUFFLE_SIZE);
		// add the new tag as the last index
		source[SHUFFLE_SIZE - 1] = tag;
		shuffle_256(source, map);
		memcpy(block + SHUFFLE_SIZE, source, SHUFFLE_SIZE);
	}
}
#endif

static inline __uint128_t update_md(__uint128_t md, uint8_t index) {
  static const __uint128_t one = 1;
  
  __uint128_t unshifted_mask = (one << index) - 1;
  __uint128_t unshifted_md = md & unshifted_mask;
  __uint128_t shifted_md = (md & ~unshifted_mask) << 1;
  return shifted_md | unshifted_md;
}

// number of 0s in the metadata is the number of tags.
static inline uint64_t get_block_free_space(__uint128_t vector) {
	uint64_t lower_word = vector & BITMASK(64);
	uint64_t higher_word = vector >> 64;
	return word_rank(lower_word) + word_rank(higher_word);
}

// Create n/log(n) blocks of log(n) slots.
// log(n) is 51 given a cache line size.
// n/51 blocks.
ququ_filter * ququ_init(uint64_t nslots) {
  ququ_filter *filter;
  
	uint64_t total_blocks = (nslots + QUQU_SLOTS_PER_BLOCK)/QUQU_SLOTS_PER_BLOCK;
  uint64_t total_size_in_bytes = sizeof(ququ_block) * total_blocks;

  filter = (ququ_filter *)malloc(sizeof(*filter) + total_size_in_bytes);
  assert(filter);
  
	filter->metadata.total_size_in_bytes = total_size_in_bytes;
	filter->metadata.nslots = total_blocks * QUQU_SLOTS_PER_BLOCK;
	filter->metadata.key_remainder_bits = 8;
	filter->metadata.range = total_blocks * QUQU_BUCKETS_PER_BLOCK * (1ULL << filter->metadata.key_remainder_bits);
	filter->metadata.nblocks = total_blocks;
	filter->metadata.nelts = 0;

	// memset to 1
	for (uint64_t i = 0; i < total_blocks; i++) {
		__uint128_t set_vector = UINT64_MAX;
		set_vector = set_vector << 64 | UINT64_MAX;
		filter->blocks[i].md = set_vector;
	}

	return filter;
}

// If the item goes in the i'th slot (starting from 0) in the block then
// find the i'th 0 in the metadata, insert a 1 after that and shift the rest
// by 1 bit.
// Insert the new tag at the end of its run and shift the rest by 1 slot.
int ququ_insert(ququ_filter * restrict filter, uint64_t hash) {
	ququ_metadata * restrict metadata           = &filter->metadata;
	ququ_block    * restrict blocks             = filter->blocks;
	uint64_t                 key_remainder_bits = metadata->key_remainder_bits;
	uint64_t                 range              = metadata->range;

	uint64_t block_index = hash >> key_remainder_bits;
	__uint128_t block_md = blocks[block_index         / QUQU_BUCKETS_PER_BLOCK].md;

	uint64_t tag = hash & BITMASK(key_remainder_bits);

	uint64_t alt_block_index = ((block_index ^ (tag * 0x5bd1e995)) % range) >> key_remainder_bits;
	__uint128_t alt_block_md = blocks[alt_block_index     / QUQU_BUCKETS_PER_BLOCK].md;

	uint64_t block_free     =	get_block_free_space(block_md);
	uint64_t alt_block_free =	get_block_free_space(alt_block_md);

	// pick the least loaded block
	if (alt_block_free > block_free) {
		block_index = alt_block_index;
		block_md = alt_block_md;
	} else if (block_free == QUQU_BUCKETS_PER_BLOCK) {
		fprintf(stderr, "ququ filter is full.");
		exit(EXIT_FAILURE);
	}

	uint64_t index = block_index / QUQU_BUCKETS_PER_BLOCK;
	uint64_t offset = block_index % QUQU_BUCKETS_PER_BLOCK;

	uint64_t select_index = select_128(block_md, offset);
	uint64_t slot_index = select_index - offset;

	/*printf("tag: %ld offset: %ld\n", tag, offset);*/
	/*print_block(filter, index);*/

#if 1
	update_tags(&blocks[index], slot_index,	tag);
#else
	update_tags(reinterpret_cast<uint8_t*>(&blocks[index]), slot_index,tag);
#endif
	blocks[index].md = update_md(block_md, select_index);

	/*print_block(filter, index);*/
	return 0;
}

static inline bool check_tags(ququ_filter * restrict filter, uint8_t tag, uint64_t block_index) {
	uint64_t index = block_index / QUQU_BUCKETS_PER_BLOCK;
	uint64_t offset = block_index % QUQU_BUCKETS_PER_BLOCK;

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
bool ququ_is_present(ququ_filter * restrict filter, uint64_t hash) {
	uint64_t tag = hash & BITMASK(filter->metadata.key_remainder_bits);
	uint64_t block_index = hash >> filter->metadata.key_remainder_bits;
	uint64_t alt_block_index = ((block_index ^ (tag * 0x5bd1e995)) %
															filter->metadata.range) >>
		filter->metadata.key_remainder_bits;

	bool ret = check_tags(filter, tag, block_index) ? true : check_tags(filter, tag,
																																	alt_block_index);
	/*if (!ret) {*/
		/*printf("tag: %ld offset: %ld\n", tag, block_index % QUQU_SLOTS_PER_BLOCK);*/
		/*print_block(filter, block_index / QUQU_SLOTS_PER_BLOCK);*/
		/*print_block(filter, alt_block_index / QUQU_SLOTS_PER_BLOCK);*/
	/*}*/
	return ret;
}

