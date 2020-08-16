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
#include <libpmemobj.h>

#ifdef __cplusplus
#define restrict __restrict__
extern "C" {
#endif

#define VALUE_BITS 0

	// metadata: 1 --> end of the run
	// Each 1 is preceded by k 0s, where k is the number of remainders in that
	// run.

	// We are using 8-bit slots.
	// One block consists of 48 8-bit slots covering 80 buckets, and 80+48 = 128 bits of metadata.
	typedef struct __attribute__ ((__packed__)) ququ_block {
		uint64_t md[2];
		uint8_t tags[48];
	} ququ_block;

	typedef struct ququ_metadata {
		uint64_t total_size_in_bytes;
		uint64_t key_remainder_bits;
		uint64_t range;
		uint64_t nblocks;
		uint64_t nelts;
		uint64_t nslots;
	} ququ_metadata;

	typedef struct ququ_filter {
		ququ_metadata metadata;
		ququ_block blocks[];
	} ququ_filter;

	PMEMobjpool * ququ_init(uint64_t nslots);

	bool ququ_insert(PMEMobjpool * pop, uint64_t hash);
	bool ququ_insert_tx(PMEMobjpool * pop, uint64_t hash);
	
	bool ququ_remove(PMEMobjpool * pop, uint64_t hash);
	bool ququ_remove_tx(PMEMobjpool * pop, uint64_t hash);

#if VALUE_BITS == 0
	bool ququ_is_present(PMEMobjpool * pop, uint64_t hash);
	bool ququ_is_present_tx(PMEMobjpool * pop, uint64_t hash);
#else
	bool ququ_is_present(ququ_filter * restrict filter, uint64_t hash, uint8_t *value);
	bool ququ_is_present_tx(ququ_filter * restrict filter, uint64_t hash, uint8_t *value);

	bool ququ_set(ququ_filter * restrict filter, uint64_t hash, uint8_t value);
	bool ququ_set_tx(ququ_filter * restrict filter, uint64_t hash, uint8_t value);
#endif

#ifdef __cplusplus
}
#endif

#endif	// _QUQU_FILTER_H_


