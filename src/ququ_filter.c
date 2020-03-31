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

#include <algorithm>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>  // portable to all x86 compilers
#include <tmmintrin.h>

#include "shuffle_matrix_256.h"
#include "shuffle_matrix_512.h"
#include "ququ_filter.h"

#define MAX_VALUE(nbits) ((1ULL << (nbits)) - 1)
#define BITMASK(nbits)                                    \
  ((nbits) == 64 ? 0xffffffffffffffff : MAX_VALUE(nbits))

#define QUQU_SLOTS_PER_BLOCK 48
#define QUQU_BUCKETS_PER_BLOCK 80
#define QUQU_CHECK_ALT 92

static inline int word_rank(uint64_t val) {
	return __builtin_popcountll(val);
	/*asm("popcnt %[val], %[val]"*/
			/*: [val] "+r" (val)*/
			/*:*/
			/*: "cc");*/
	/*return val;*/
}

static uint64_t pre_one[64 + 128] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   1ULL << 0, 1ULL << 1, 1ULL << 2, 1ULL << 3, 1ULL << 4, 1ULL << 5, 1ULL << 6, 1ULL << 7, 1ULL << 8, 1ULL << 9,
   1ULL << 10, 1ULL << 11, 1ULL << 12, 1ULL << 13, 1ULL << 14, 1ULL << 15, 1ULL << 16, 1ULL << 17, 1ULL << 18, 1ULL << 19, 
   1ULL << 20, 1ULL << 21, 1ULL << 22, 1ULL << 23, 1ULL << 24, 1ULL << 25, 1ULL << 26, 1ULL << 27, 1ULL << 28, 1ULL << 29, 
   1ULL << 30, 1ULL << 31, 1ULL << 32, 1ULL << 33, 1ULL << 34, 1ULL << 35, 1ULL << 36, 1ULL << 37, 1ULL << 38, 1ULL << 39, 
   1ULL << 40, 1ULL << 41, 1ULL << 42, 1ULL << 43, 1ULL << 44, 1ULL << 45, 1ULL << 46, 1ULL << 47, 1ULL << 48, 1ULL << 49, 
   1ULL << 50, 1ULL << 51, 1ULL << 52, 1ULL << 53, 1ULL << 54, 1ULL << 55, 1ULL << 56, 1ULL << 57, 1ULL << 58, 1ULL << 59, 
   1ULL << 60, 1ULL << 61, 1ULL << 62, 1ULL << 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

static uint64_t *one = pre_one + 64;

// Returns the position of the rank'th 1.  (rank = 0 returns the 1st 1)
// Returns 64 if there are fewer than rank+1 1s.
static inline uint64_t word_select(uint64_t val, int rank) {
	val = _pdep_u64(one[rank], val);
	return _tzcnt_u64(val);
	/*asm("pdep %[val], %[mask], %[val]"*/
			/*: [val] "+r" (val)*/
			/*: [mask] "r" (i));*/
	/*asm("tzcnt %[bit], %[index]"*/
			/*: [index] "=r" (i)*/
			/*: [bit] "g" (val)*/
			/*: "cc");*/
	/*return i;*/
}

// select(vec, 0) -> -1
// select(vec, i) -> 128, if i > popcnt(vec)
static inline int64_t select_128_old(__uint128_t vector, uint64_t rank) {
	uint64_t lower_word = vector & 0xffffffffffffffff;
        uint64_t lower_pdep = _pdep_u64(one[rank], lower_word);
        //uint64_t lower_select = word_select(lower_word, rank);
        if (lower_pdep != 0) {
           //assert(rank < word_rank(lower_word));
           return _tzcnt_u64(lower_pdep);
	}
	rank = rank - word_rank(lower_word);
	uint64_t higher_word = vector >> 64;
        return word_select(higher_word, rank) + 64;
}

static inline uint64_t lookup_128(uint64_t *vector, uint64_t rank) {
	uint64_t lower_word = vector[0];
	uint64_t lower_rank = word_rank(lower_word);
        uint64_t lower_return = _pdep_u64(one[rank], lower_word) >> rank << sizeof(__uint128_t);
        int64_t higher_rank = (int64_t)rank - lower_rank;
	uint64_t higher_word = vector[1];
        uint64_t higher_return = _pdep_u64(one[higher_rank], higher_word);
        higher_return <<= (64 + sizeof(__uint128_t) - rank);
        return lower_return + higher_return;
}

static inline int64_t select_128(uint64_t *vector, uint64_t rank) {
        return _tzcnt_u64(lookup_128(vector, rank));
}

//assumes little endian
void print_bits(__uint128_t num, int numbits)
{
  int i;
  for (i = 0 ; i < numbits; i++) {
    if (i != 0 && i % 8 == 0) {
       printf(":");
    }
    printf("%d", ((num >> i) & 1) == 1);
  }
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

#if 0
static inline void update_tags(ququ_block * restrict block, uint8_t index, uint8_t tag) {
	memmove(&block->tags[index + 1], &block->tags[index], sizeof(block->tags) / sizeof(block->tags[0]) - index - 1);
	block->tags[index] = tag;
}

#else

static inline void update_tags_512(ququ_block * restrict block, uint8_t index,
																	 uint8_t tag) {
	block->tags[47] = tag;	// add tag at the end

        __m512i vector = _mm512_loadu_si512(reinterpret_cast<__m512i*>(block));
	vector = _mm512_permutexvar_epi8(SHUFFLE[index], vector);
	_mm512_storeu_si512(reinterpret_cast<__m512i*>(block), vector);
}

const __m256i K0 = _mm256_setr_epi8( 
		0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 
		0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0); 

const __m256i K1 = _mm256_setr_epi8( 
		0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 
		0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70, 0x70); 

const __m256i K[] = {K0, K1};

#define QUQU_CHECK_ALT 92
inline __m256i cross_lane_shuffle(const __m256i & value, const __m256i & shuffle) 
{ 
 	 return _mm256_or_si256(_mm256_shuffle_epi8(value, _mm256_add_epi8(shuffle, K[0])), 
 			 _mm256_shuffle_epi8(_mm256_permute4x64_epi64(value, 0x4E), _mm256_add_epi8(shuffle, K[1]))); 
} 

#define SHUFFLE_SIZE 32

void shuffle_256(uint8_t * restrict source, __m256i shuffle) {
  __m256i vector = _mm256_loadu_si256(reinterpret_cast<__m256i*>(source)); 

  vector = cross_lane_shuffle(vector, shuffle); 
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(source), vector); 
} 

static inline void update_tags(uint8_t * restrict block, uint8_t index, uint8_t tag) {
	index = index + sizeof(__uint128_t);	// offset index based on md field.
	block[63] = tag;	// add tag at the end
	shuffle_256(block + SHUFFLE_SIZE, RM[index]); // right block shuffle
	if (index < SHUFFLE_SIZE) {		// if index lies in the left block
		std::swap(block[31], block[32]);	// move tag to the end of left block
		shuffle_256(block, LM[index]);	// shuffle left block
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
	uint64_t lower_word = vector & 0xffffffffffffffff;
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
	printf("Size: %ld\n",total_size_in_bytes);
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
	uint64_t tag = hash & 0xff;
	uint64_t block_free     =	get_block_free_space(block_md);
	uint64_t alt_block_index = ((hash ^ (tag * 0x5bd1e995)) % range) >> key_remainder_bits;

	__builtin_prefetch(&blocks[alt_block_index / QUQU_BUCKETS_PER_BLOCK]);

	if (block_free < QUQU_CHECK_ALT) {
		__uint128_t alt_block_md = blocks[alt_block_index     / QUQU_BUCKETS_PER_BLOCK].md;
		uint64_t alt_block_free =	get_block_free_space(alt_block_md);
		// pick the least loaded block
		if (alt_block_free > block_free) {
			block_index = alt_block_index;
			block_md = alt_block_md;
		} else if (block_free == QUQU_BUCKETS_PER_BLOCK) {
			fprintf(stderr, "ququ filter is full.");
			exit(EXIT_FAILURE);
		}
	}

	uint64_t index = block_index / QUQU_BUCKETS_PER_BLOCK;
	uint64_t offset = block_index % QUQU_BUCKETS_PER_BLOCK;

	uint64_t slot_index = select_128((uint64_t *)&block_md, offset);
        uint64_t select_index = slot_index + offset - sizeof(__uint128_t);
	/*printf("index: %ld tag: %ld offset: %ld\n", index, tag, offset);*/
	/*print_block(filter, index);*/

#if 0
	update_tags(&blocks[index], slot_index,	tag);
#else
	//update_tags(reinterpret_cast<uint8_t*>(&blocks[index]), slot_index,tag);
	update_tags_512(&blocks[index], slot_index,tag);
#endif
	blocks[index].md = update_md(block_md, select_index);
        
	/*print_block(filter, index);*/
	return 0;
}

static inline bool check_tags(ququ_filter * restrict filter, uint8_t tag, uint64_t block_index) {
	uint64_t index = block_index / QUQU_BUCKETS_PER_BLOCK;
	uint64_t offset = block_index % QUQU_BUCKETS_PER_BLOCK;

	__m512i bcast = _mm512_set1_epi8(tag);
	__m512i block = _mm512_loadu_si512(reinterpret_cast<__m512i*>(&filter->blocks[index]));
	volatile __mmask64 result = _mm512_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);

	if (result == 0) {
		// no matching tags, can bail
		return false;
	}

        uint64_t start = offset != 0 ? lookup_128((uint64_t *)&filter->blocks[index].md, offset - 1) : one[0] << sizeof(__uint128_t);
        uint64_t end = lookup_128((uint64_t *)&filter->blocks[index].md, offset);
        uint64_t mask = end - start;
	return (mask & result) != 0;
}

// If the item goes in the i'th slot (starting from 0) in the block then
// select(i) - i is the slot index for the end of the run.
bool ququ_is_present(ququ_filter * restrict filter, uint64_t hash) {
	ququ_metadata * restrict metadata           = &filter->metadata;
	ququ_block    * restrict blocks             = filter->blocks;
	uint64_t                 key_remainder_bits = metadata->key_remainder_bits;
	uint64_t                 range              = metadata->range;

	uint64_t block_index = hash >> key_remainder_bits;
	__uint128_t block_md = blocks[block_index         / QUQU_BUCKETS_PER_BLOCK].md;
	uint64_t tag = hash & 0xff;
	uint64_t block_free     =	get_block_free_space(block_md);
	uint64_t alt_block_index = ((hash ^ (tag * 0x5bd1e995)) % range) >> key_remainder_bits;

	__builtin_prefetch(&filter->blocks[alt_block_index / QUQU_BUCKETS_PER_BLOCK]);
        
//	if (block_free < QUQU_CHECK_ALT) {
	    return check_tags(filter, tag, block_index) ? true : check_tags(filter, tag, alt_block_index);
//        } else {
//           return check_tags(filter, tag, block_index); 
//       }

	/*if (!ret) {*/
		/*printf("tag: %ld offset: %ld\n", tag, block_index % QUQU_SLOTS_PER_BLOCK);*/
		/*print_block(filter, block_index / QUQU_SLOTS_PER_BLOCK);*/
		/*print_block(filter, alt_block_index / QUQU_SLOTS_PER_BLOCK);*/
	/*}*/
}

