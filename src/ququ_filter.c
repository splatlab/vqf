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
	if (rank == 0) {
		return -1;
	} else {
		uint64_t lower_word = vector & BITMASK(64);
		uint64_t lower_rank = word_rank(lower_word);
		if (lower_rank > rank) {
			return word_select(lower_word, rank);
		} else {
			rank = rank - lower_rank;
			uint64_t higher_word = vector >> 64;
			if ((uint64_t)word_rank(lower_word) > rank) {
				return word_select(higher_word, rank);
			} else {
				return 128;
			}
		}
	}
}

#define SHUFFLE_SIZE 32

void shuffle_256(uint8_t *source, uint8_t *map) {
	__m256i vector = _mm256_loadu_si256(reinterpret_cast<__m256i*>(source));
	__m256i shuffle = _mm256_loadu_si256(reinterpret_cast<__m256i*>(map));

	vector = _mm256_shuffle_epi8(vector, shuffle);
	_mm256_storeu_si256(reinterpret_cast<__m256i*>(source), vector);
}

void update_tags(uint8_t *block, uint8_t index, uint8_t tag) {
	index = index + 13;		// offset index based on the md size
	if (index < SHUFFLE_SIZE) {	// change in the first 32-bytes. Must move both halves.
		/* Move first 32-bytes */
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
		shuffle_256(source, map);
		memcpy(block, source, SHUFFLE_SIZE);

		/* move second 32-bytes */
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
		memcpy(source, block, SHUFFLE_SIZE);
		// add the new tag as the last index
		source[SHUFFLE_SIZE - 1] = tag;
		shuffle_256(source, map);
		memcpy(block, source, SHUFFLE_SIZE);
	}
}

__uint128_t update_md(__uint128_t md, uint8_t index, uint8_t bit) {
	__uint128_t updated_md = md;
	if (!bit) {
		__uint128_t mask = 1ULL << index;
		updated_md = updated_md & ~mask;
	} else {
		__uint128_t mask = 1ULL << index;
		updated_md = updated_md | mask;
	}
	return updated_md;
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
		perror("ququ filter is full.");
		exit(EXIT_FAILURE);
	}
	// pick the least loaded block
	if (alt_load < primary_load) {
		block_index = alt_block_index;
	}

	uint64_t index = block_index / QUQU_SLOTS_PER_BLOCK;
	uint64_t offset = block_index % QUQU_SLOTS_PER_BLOCK;

	uint64_t select_index = select_128(filter->blocks[index].md, offset + 1);
	uint64_t slot_index = select_index - offset;
	update_tags(reinterpret_cast<uint8_t*>(&filter->blocks[index]), slot_index,
							tag);
	filter->blocks[block_index].md = update_md(filter->blocks[index].md,
																						 select_index, 0);
	return 0;
}

bool check_tags(ququ_filter *filter, uint8_t tag, uint64_t block_index) {
	uint64_t block_offset = block_index % QUQU_SLOTS_PER_BLOCK;
	uint64_t start = select_128(filter->blocks[block_index].md, block_offset) -
		block_offset + 1;
	uint64_t end = select_128(filter->blocks[block_index].md, block_offset + 1)
		- block_offset;

	for (uint64_t i = start; i < end; i++) {
		if (tag == filter->blocks[block_index].tags[i])
			return true;
	}
	return false;
}

// If the item goes in the i'th slot (starting from 0) in the block then
// select(i) - i is the slot index for the end of the run.
bool ququ_is_present(ququ_filter *filter, __uint128_t hash) {
	uint64_t tag = hash & BITMASK(filter->metadata->key_remainder_bits);
	uint64_t block_index = hash >> filter->metadata->key_remainder_bits;
	uint64_t alt_block_index = block_index ^ (tag * 0x5bd1e995) >>
		filter->metadata->key_remainder_bits;

	return check_tags(filter, tag, block_index) ? true : check_tags(filter, tag,
																																	alt_block_index);
}

