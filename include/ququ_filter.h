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
extern "C" {
#endif

	// We are using 8-bit slots.
	// One block consists of 48 8-bit slots and 64 2-bit metadata.
	typedef struct __attribute__ ((__packed__)) ququ_block {
		uint64_t metadata[2];
		uint8_t tags[48];
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
		uint64_t noccupied_slots;
	} ququ_metadata;

	typedef struct ququ_filter {
		ququ_metadata *metadata;
		ququ_block *blocks;
	} ququ_filter;

	int ququ_init(ququ_filter *filter, uint64_t nslots, uint64_t keybits);

	int ququ_insert(ququ_filter *filter, uint64_t item);

	bool ququ_is_present(ququ_filter *filter, uint64_t item);

#ifdef __cplusplus
}
#endif

#endif	// _QUQU_FILTER_H_


