/*
 * ============================================================================
 *
 *       Filename:  ququ_filter.h
 *
 *         Author:  Prashant Pandey (), ppandey2@cs.cmu.edu
 *   Organization:  Carnegie Mellon University
 *
 * ============================================================================
 */

#ifndef _QUQU_FILTER_H_
#define _QUQU_FILTER_H_

#include <inttypes.h>
#include <stdbool.h>

#ifdef __cplusplus
#define restrict __restrict__
extern "C" {
#endif

	// metadata: 1 --> end of the run
	// Each 1 is preceded by k 0s, where k is the number of remainders in that
	// run.

	// We are using 8-bit slots.
	// One block consists of 51 8-bit slots and 51 2-bit metadata.
	typedef struct __attribute__ ((__packed__)) ququ_block {
		__uint128_t md:102;
		__uint8_t reserved:2;
		uint8_t tags[51];
	} ququ_block;

	typedef struct ququ_metadata {
		uint64_t total_size_in_bytes;
		uint32_t seed;
		uint64_t key_bits;
		uint64_t key_remainder_bits;
		__uint128_t range;
		uint64_t nblocks;
		uint64_t nelts;
		uint64_t nslots;
	} ququ_metadata;

	typedef struct ququ_filter {
		ququ_metadata * restrict metadata;
		ququ_block * restrict blocks;
	} ququ_filter;

	int ququ_init(ququ_filter * restrict filter, uint64_t nslots);

	int ququ_insert(ququ_filter * restrict filter, __uint128_t hash);

	bool ququ_is_present(ququ_filter * restrict filter, __uint128_t hash);

#ifdef __cplusplus
}
#endif

#endif	// _QUQU_FILTER_H_


