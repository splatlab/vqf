/*
 * ============================================================================
 *
 *       Filename:  main.cc
 *
 *         Author:  Prashant Pandey (), ppandey2@cs.cmu.edu
 *   Organization:  Carnegie Mellon University
 *
 * ============================================================================
 */

#include <iostream>
#include <stdint.h>
#include <immintrin.h>  // portable to all x86 compilers
#include <tmmintrin.h>
#include <openssl/rand.h>

#include "ququ_filter.h"

int main(int argc, char **argv)
{
	if (argc < 2) {
		fprintf(stderr, "Please specify the log of the number of slots in the CQF.\n");
		exit(1);
	}
	uint64_t qbits = atoi(argv[1]);
	uint64_t nslots = (1ULL << qbits);
	uint64_t nvals = 75*nslots/100;
	uint64_t *vals;

	ququ_filter filter;	

	/* initialize ququ filter */
	if (ququ_init(&filter, nslots)) {
		perror("Can't allocate ququ filter.");
		exit(EXIT_FAILURE);
	}

	/* Generate random values */
	vals = (uint64_t*)malloc(nvals*sizeof(vals[0]));
	RAND_bytes((unsigned char *)vals, sizeof(*vals) * nvals);
	srand(0);
	for (uint64_t i = 0; i < nvals; i++) {
		vals[i] = (1 * vals[i]) % filter.metadata->range;
	}

	/* Insert hashes in the ququ filter */
	for (uint64_t i = 0; i < nvals; i++) {
		if (ququ_insert(&filter, vals[i])) {
			perror("Insertion failed");
			exit(EXIT_FAILURE);
		}
	}

	for (uint64_t i = 0; i < nvals; i++) {
		if (!ququ_insert(&filter, vals[i])) {
			perror("Lookup failed");
			exit(EXIT_FAILURE);
		}
	}

#if 0
	uint8_t source[SIZE];
	uint8_t order[SIZE];
	for (uint8_t i = 0; i < SIZE; i++) {
		source[i] = i;
		order[i] = i - 1;
	}

	std::cout << "vector before shuffle: \n";
	for (uint8_t i = 0; i < SIZE; i++)
		std::cout << (uint32_t)source[i] << " ";
	std::cout << "\n";

	__m256i vector = _mm256_loadu_si256(reinterpret_cast<__m256i*>(source));
	__m256i shuffle = _mm256_loadu_si256(reinterpret_cast<__m256i*>(order));

	vector = _mm256_shuffle_epi8(vector, shuffle);
	_mm256_storeu_si256(reinterpret_cast<__m256i*>(source), vector);

	std::cout << "vector after shuffle: \n";
	for (uint8_t i = 0; i < SIZE; i++)
		std::cout << (uint32_t)source[i] << " ";
	std::cout << "\n";
#endif
	return 0;
}
