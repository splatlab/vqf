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

#include <stdint.h>
#include <stdio.h>
#include <immintrin.h>  // portable to all x86 compilers
#include <tmmintrin.h>
#include <openssl/rand.h>
#include <sys/time.h>

#include "ququ_filter.h"

/* Print elapsed time using the start and end timeval */
void print_time_elapsed(const char* desc, struct timeval* start, struct
												timeval* end)
{
	struct timeval elapsed;
	if (start->tv_usec > end->tv_usec) {
		end->tv_usec += 1000000;
		end->tv_sec--;
	}
	elapsed.tv_usec = end->tv_usec - start->tv_usec;
	elapsed.tv_sec = end->tv_sec - start->tv_sec;
	float time_elapsed = (elapsed.tv_sec * 1000000 + elapsed.tv_usec)/1000000.f;
	printf("%s Total Time Elapsed: %f seconds", desc, time_elapsed);
}

int main(int argc, char **argv)
{
#if 1
	if (argc < 2) {
		fprintf(stderr, "Please specify the log of the number of slots in the CQF.\n");
		exit(1);
	}
	uint64_t qbits = atoi(argv[1]);
	uint64_t nslots = (1ULL << qbits);
	uint64_t nvals = 85*nslots/100;
	uint64_t *vals;

	ququ_filter *filter;	

	/* initialize ququ filter */
	if ((filter = ququ_init(nslots)) == NULL) {
		fprintf(stderr, "Can't allocate ququ filter.");
		exit(EXIT_FAILURE);
	}

	/* Generate random values */
	vals = (uint64_t*)malloc(nvals*sizeof(vals[0]));
	RAND_bytes((unsigned char *)vals, sizeof(*vals) * nvals);
	srand(0);
	for (uint64_t i = 0; i < nvals; i++) {
		//vals[i] = rand() % filter.metadata->range;
		vals[i] = (1 * vals[i]) % filter->metadata.range;
	}

	struct timeval start, end;
	struct timezone tzp;

	gettimeofday(&start, &tzp);
	/* Insert hashes in the ququ filter */
	for (uint64_t i = 0; i < nvals; i++) {
		if (ququ_insert(filter, vals[i]) < 0) {
			fprintf(stderr, "Insertion failed");
			exit(EXIT_FAILURE);
		}
		//if (!ququ_is_present(&filter, vals[i])) {
			//fprintf(stderr, "Lookup failed for %ld", vals[i]);
			//exit(EXIT_FAILURE);
		//}
	}
	gettimeofday(&end, &tzp);
	print_time_elapsed("Insertion:", &start, &end);
	puts("");
	gettimeofday(&start, &tzp);
	/* Lookup hashes in the ququ filter */
	for (uint64_t i = 0; i < nvals; i++) {
		if (!ququ_is_present(filter, vals[i])) {
			fprintf(stderr, "Lookup failed for %ld", vals[i]);
			exit(EXIT_FAILURE);
		}
	}
	gettimeofday(&end, &tzp);
	print_time_elapsed("Lookup:", &start, &end);

#else
#define SIZE 32
	uint8_t source[SIZE];
	uint8_t order[SIZE];
	for (uint8_t i = 0; i < SIZE; i++) {
		source[i] = i;
		order[i] = SIZE-i-1;
	}

	std::cout << "order vector: \n";
	for (uint8_t i = 0; i < SIZE; i++)
		std::cout << (uint32_t)order[i] << " ";
	std::cout << "\n";

	std::cout << "vector before shuffle: \n";
	for (uint8_t i = 0; i < SIZE; i++)
		std::cout << (uint32_t)source[i] << " ";
	std::cout << "\n";

	__m256i vector = _mm256_loadu_si256(reinterpret_cast<__m256i*>(source));
	__m256i shuffle = _mm256_loadu_si256(reinterpret_cast<__m256i*>(order));

	vector = _mm256_shuffle_epi8(vector, shuffle);
	//vector = Shuffle(vector, shuffle);
	_mm256_storeu_si256(reinterpret_cast<__m256i*>(source), vector);

	std::cout << "vector after shuffle: \n";
	for (uint8_t i = 0; i < SIZE; i++)
		std::cout << (uint32_t)source[i] << " ";
	std::cout << "\n";
#endif
	return 0;
}
