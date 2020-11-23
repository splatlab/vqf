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
#include <stdio.h>
#include <math.h>
#include <immintrin.h>  // portable to all x86 compilers
#include <tmmintrin.h>
#include <openssl/rand.h>
#include <sys/time.h>

#include <set>

#include "ququ_filter.h"

extern __m512i SHUFFLE [];
extern __m512i SHUFFLE_REMOVE [];
extern __m512i SHUFFLE16 [];
extern __m512i SHUFFLE_REMOVE16 [];

uint64_t tv2usec(struct timeval *tv) {
  return 1000000 * tv->tv_sec + tv->tv_usec;
}

/* Print elapsed time using the start and end timeval */
void print_time_elapsed(const char* desc, struct timeval* start, struct
												timeval* end, uint64_t ops, const char *opname)
{
  uint64_t elapsed_usecs = tv2usec(end) - tv2usec(start);
	printf("%s Total Time Elapsed: %f seconds", desc, 1.0*elapsed_usecs / 1000000);
  if (ops) {
    printf(" (%f nanoseconds/%s)", 1000.0 * elapsed_usecs / ops, opname);
  }
  printf("\n");
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
	uint64_t *other_vals;

	ququ_filter *filter;	

	/* initialize ququ filter */
	if ((filter = ququ_init(nslots)) == NULL) {
		fprintf(stderr, "Can't allocate ququ filter.");
		exit(EXIT_FAILURE);
	}

	/* Generate random values */
	vals = (uint64_t*)malloc(nvals*sizeof(vals[0]));
	RAND_bytes((unsigned char *)vals, sizeof(*vals) * nvals);
	other_vals = (uint64_t*)malloc(nvals*sizeof(other_vals[0]));
	RAND_bytes((unsigned char *)other_vals, sizeof(*other_vals) * nvals);
	for (uint64_t i = 0; i < nvals; i++) {
		vals[i] = (1 * vals[i]) % filter->metadata.range;
		other_vals[i] = (1 * other_vals[i]) % filter->metadata.range;
	}
/*
        srand(0);
        for (uint32_t i = 0; i < nvals; i++) {
           vals[i] = (rand() % filter->metadata.range);
        }
*/
	struct timeval start, end;
	struct timezone tzp;

	gettimeofday(&start, &tzp);
	/* Insert hashes in the ququ filter */
	for (uint64_t i = 0; i < nvals; i++) {
		if (!ququ_insert(filter, vals[i])) {
			fprintf(stderr, "Insertion failed");
			exit(EXIT_FAILURE);
		}
         }
	gettimeofday(&end, &tzp);
	print_time_elapsed("Insertion time", &start, &end, nvals, "insert");
//	puts("");
	gettimeofday(&start, &tzp);
	for (uint64_t i = 0; i < nvals; i++) {
#if VALUE_BITS == 0
		if (!ququ_is_present(filter, vals[i])) {
#else
                uint8_t value;
		if (!ququ_is_present(filter, vals[i], &value)) {
#endif
			fprintf(stderr, "Lookup failed for %ld", vals[i]);
			exit(EXIT_FAILURE);
		}
	}
	gettimeofday(&end, &tzp);
	print_time_elapsed("Lookup time", &start, &end, nvals, "successful lookup");
	gettimeofday(&start, &tzp);
  uint64_t nfps = 0;
	/* Lookup hashes in the ququ filter */
	for (uint64_t i = 0; i < nvals; i++) {
#if VALUE_BITS == 0
		if (ququ_is_present(filter, other_vals[i])) {
#else
                uint8_t value;
		if (ququ_is_present(filter, other_vals[i], &value)) {
#endif
      nfps++;
		}
	}
	gettimeofday(&end, &tzp);
	print_time_elapsed("Random lookup:", &start, &end, nvals, "random lookup");
        printf("%lu/%lu positives\n"
         "FP rate: 1/%f\n",
         nfps, nvals,
         1.0 * nvals / nfps);

        //fprintf(stdout, "Checking ququ_remove\n");

	gettimeofday(&start, &tzp);
	for (uint64_t i = 0; i < nvals; i++) {
           ququ_remove(filter, vals[i]);
           //if (ququ_is_present(filter, vals[i])) {
           //   fprintf(stderr, "Lookup true after deletion for %ld", vals[i]);
           //   exit(EXIT_FAILURE);
           //}
        }
	gettimeofday(&end, &tzp);
	print_time_elapsed("Remove time", &start, &end, nvals, "remove");

#if 0
	//* Generate random values */
	vals = (uint64_t*)malloc(nvals*sizeof(vals[0]));
        uint64_t nbytes = sizeof(*vals) * nvals;
        uint8_t *ptr = (uint8_t *)vals;
	while (nbytes > (1ULL << 30)) {
		RAND_bytes(ptr, 1ULL << 30);
		ptr += (1ULL << 30);
		nbytes -= (1ULL << 30);
	}
	RAND_bytes(ptr, nbytes);
	
	//srand(0);
	for (uint64_t i = 0; i < nvals; i++) {
		//vals[i] = rand() % filter->metadata.range;
		vals[i] = (1 * vals[i]) % filter->metadata.range;
	}

	struct timeval start, end;
	struct timezone tzp;

	gettimeofday(&start, &tzp);
	/* Insert hashes in the ququ filter */
	for (uint64_t i = 0; i < nvals; i++) {
		if (!ququ_insert(filter, vals[i])) {
			fprintf(stderr, "Insertion failed");
			exit(EXIT_FAILURE);
		}
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
	puts("");
#endif

#else
#define SIZE 64

        srand(0);
#if 1
	uint8_t source[SIZE];
	uint8_t order[SIZE];

        for (int idx = 16; idx < 17; idx++) {
           std::cout << "index: " << idx << "\n"; 
           for (uint8_t i = 0; i < SIZE; i++) {
#if 1
              if (i == SIZE - 2)
                 source[i] = 111;
              else if (i == SIZE - 1)
                 source[i] = 132;
               else 
                  source[i] = 255;
#else
               source[i] = i;
#endif
           }

           _mm512_storeu_si512(reinterpret_cast<__m512i*>(order), SHUFFLE16[idx]);
           std::cout << "order vector: \n";
           for (uint8_t i = 0; i < SIZE; i++)
              std::cout << (uint32_t)order[i] << " ";
           std::cout << "\n";

           std::cout << "vector before shuffle: \n";
           for (uint8_t i = 0; i < SIZE; i++)
              std::cout << (uint32_t)source[i] << " ";
           std::cout << "\n";

           __m512i vector = _mm512_loadu_si512(reinterpret_cast<__m512i*>(source));
           __m512i shuffle = _mm512_loadu_si512(reinterpret_cast<__m512i*>(order));

           vector = _mm512_permutexvar_epi8(shuffle, vector);
           //vector = _mm512_shuffle_epi8(vector, shuffle);
           //vector = Shuffle(vector, shuffle);
           _mm512_storeu_si512(reinterpret_cast<__m512i*>(source), vector);

           std::cout << "vector after shuffle: \n";
           for (uint8_t i = 0; i < SIZE; i++)
              std::cout << (uint32_t)source[i] << " ";
           std::cout << "\n";
 
           _mm512_storeu_si512(reinterpret_cast<__m512i*>(order), SHUFFLE_REMOVE16[idx]);
           std::cout << "order vector: \n";
           for (uint8_t i = 0; i < SIZE; i++)
              std::cout << (uint32_t)order[i] << " ";
           std::cout << "\n";
           shuffle = _mm512_loadu_si512(reinterpret_cast<__m512i*>(order));
          
           vector = _mm512_permutexvar_epi8(shuffle, vector);
           _mm512_storeu_si512(reinterpret_cast<__m512i*>(source), vector);

           std::cout << "vector after shuffle: \n";
           for (uint8_t i = 0; i < SIZE; i++)
              std::cout << (uint32_t)source[i] << " ";
           std::cout << "\n";
        }
#endif

#endif
	return 0;
}
