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

// Returns the position of the rank'th 1.  (rank = 0 returns the 1st 1)
// Returns 64 if there are fewer than rank+1 1s.
static inline uint64_t word_select(uint64_t val, int rank) {
	uint64_t i = 1ULL << rank;
	val = _pdep_u64(i, val);
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
static inline int64_t select_128(__uint128_t vector, uint64_t rank) {
	uint64_t lower_word = vector & 0xffffffffffffffff;
	uint64_t lower_rank = word_rank(lower_word);
	if (lower_rank > rank) {
		return word_select(lower_word, rank);
	} else {
		rank = rank - lower_rank;
		uint64_t higher_word = vector >> 64;
                return word_select(higher_word, rank) + 64;
	}
}

static inline int64_t word_bsf(uint64_t val)
{
   asm("bsfq %[val], %[val]"
         : [val] "+r" (val)
         :
         : "cc");
   return val;
}

static inline int bsf_128(__uint128_t u, int index) {
   static const __uint128_t one = 1;
   __uint128_t unshifted_mask = (one << index) - 1;
   u = u & ~unshifted_mask;
   uint64_t hi = u >> 64;
   uint64_t lo = u;
   int lo_eq_0 = (lo == 0); 
   uint64_t hi_or_lo = lo_eq_0 ? hi : lo;
   int bsf_hi_or_lo = word_bsf(hi_or_lo);
   return bsf_hi_or_lo + (lo_eq_0 << 6);
}

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

#if 0
static inline void update_tags(ququ_block * restrict block, uint8_t index, uint8_t tag) {
	memmove(&block->tags[index + 1], &block->tags[index], sizeof(block->tags) / sizeof(block->tags[0]) - index - 1);
	block->tags[index] = tag;
}

#else

static inline void update_tags_512(ququ_block * restrict block, uint8_t index,
																	 uint8_t tag) {
	index = index + sizeof(__uint128_t);	// offset index based on md field.
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

	uint64_t select_index = select_128(block_md, offset);
	uint64_t slot_index = select_index - offset;

	/*printf("index: %ld tag: %ld offset: %ld\n", index, tag, offset);*/
	/*print_block(filter, index);*/

#if 0
	update_tags(&blocks[index], slot_index,	tag);
#else
	//update_tags(reinterpret_cast<uint8_t*>(&blocks[index]), slot_index,tag);
	update_tags_512(&blocks[index], slot_index,tag);
#endif
	blocks[index].md = update_md(block_md, select_index);
        
	//print_block(filter, index);
	return 0;
}

static inline bool check_tags(ququ_filter * restrict filter, uint8_t tag, uint64_t block_index) {
	uint64_t index = block_index / QUQU_BUCKETS_PER_BLOCK;
	uint64_t offset = block_index % QUQU_BUCKETS_PER_BLOCK;

	__m512i bcast = _mm512_set1_epi8(tag);
	__m512i block = _mm512_loadu_si512(reinterpret_cast<__m512i*>(&filter->blocks[index]));
	__mmask64 result = _mm512_cmp_epi8_mask(bcast, block, _MM_CMPINT_EQ);

	if (result == 0) {
		// no matching tags, can bail
		return false;
	}

	uint64_t start, end;
        uint64_t select1;
	if (offset == 0) {
		start = 0;
                select1 = 0;
	} else {
		select1 = select_128(filter->blocks[index].md, offset - 1);
		start = select1 - offset + 1;
	}
//        uint64_t select2_new = bsf_128(filter->blocks[index].md, select1+1);
	uint64_t select2 = select_128(filter->blocks[index].md, offset);
//        if (select2 != select2_new) {
//           printf("Old End: %ld New End: %ld Index: %ld\n", select2, select2_new, select1+1);
//	    print_block(filter, index);
//            exit(1);
//        }
        end = select2 - offset;
	uint64_t mask = ((1ULL << end) - (1ULL << start)) << sizeof(__uint128_t);
	//printf("0x%lx 0x%lx\n", result, mask);
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
        
	if (block_free < QUQU_CHECK_ALT) {
	    return check_tags(filter, tag, block_index) ? true : check_tags(filter, tag, alt_block_index);
        } else {
           return check_tags(filter, tag, block_index); 
       }

	/*if (!ret) {*/
		/*printf("tag: %ld offset: %ld\n", tag, block_index % QUQU_SLOTS_PER_BLOCK);*/
		/*print_block(filter, block_index / QUQU_SLOTS_PER_BLOCK);*/
		/*print_block(filter, alt_block_index / QUQU_SLOTS_PER_BLOCK);*/
	/*}*/
}

