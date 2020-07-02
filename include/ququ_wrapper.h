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

POBJ_LAYOUT_BEGIN(ququ);
POBJ_LAYOUT_TOID(ququ, ququ_metadata);
POBJ_LAYOUT_TOID(ququ, ququ_block);
POBJ_LAYOUT_END(ququ);

PMEMobjpool * pop;

inline int q_init(uint64_t nbits)
{
	uint64_t nslots = (1ULL << nbits);
	pop = ququ_init(nslots);
	return 0;
}

inline int q_insert(__uint128_t val)
{
	if (!ququ_insert(pop, val))
		return 0;
	return 1;
}

inline int q_lookup(__uint128_t val)
{
	if (!ququ_is_present(pop, val))
		return 0;
	return 1;
}

inline int q_remove(__uint128_t val)
{
	if (!ququ_remove(pop, val))
		return 0;
	return 1;
}

inline __uint128_t q_range()
{
	return (D_RO(POBJ_FIRST(pop, ququ_metadata)))->range;
}

inline int q_destroy()
{
	return 0;
}

#endif
