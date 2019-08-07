/*
 * ============================================================================
 *
 *       Filename:  ququ_wrapper.h
 *
 *         Author:  Prashant Pandey (), ppandey2@cs.cmu.edu
 *   Organization:  Carnegie Mellon University
 *
 * ============================================================================
 */

#ifndef QUQU_WRAPPER_H
#define QUQU_WRAPPER_H

#include "ququ_filter.h"

ququ_filter q_filter;


inline int q_init(uint64_t nbits)
{
	uint64_t nslots = (1ULL << nbits);
	ququ_init(&q_filter, nslots);
	return 0;
}

inline int q_insert(__uint128_t val)
{
	if (ququ_insert(&q_filter, val) < 0)
		return 0;
	return 1;
}

inline int q_lookup(__uint128_t val)
{
	if (!ququ_is_present(&q_filter, val))
		return 0;
	return 1;
}

inline __uint128_t q_range()
{
	return q_filter.metadata->range;
}

inline int q_destroy()
{
	return 0;
}

#endif
