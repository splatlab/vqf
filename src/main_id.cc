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

#include "vqf_filter.h"

#define ITR 100000000 

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
   if (argc < 2) {
      fprintf(stderr, "Please specify the log of the number of slots in the CQF.\n");
      exit(1);
   }
   uint64_t qbits = atoi(argv[1]);
   uint64_t nslots = (1ULL << qbits);
   uint64_t nvals = 85*nslots/100;
   uint64_t *vals;
   uint64_t *other_vals;

   vqf_filter *filter;	

   /* initialize vqf filter */
   if ((filter = vqf_init(nslots)) == NULL) {
      fprintf(stderr, "Can't allocate vqf filter.");
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

   struct timeval start, end;
   struct timezone tzp;

   gettimeofday(&start, &tzp);
   /* Insert hashes in the vqf filter */
   for (uint64_t i = 0; i < nvals; i++) {
      if (!vqf_insert(filter, vals[i])) {
         fprintf(stderr, "Insertion failed");
         exit(EXIT_FAILURE);
      }
   }
   gettimeofday(&end, &tzp);
   print_time_elapsed("Insertion time", &start, &end, nvals, "insert");

   std::cout << "Starting workload\n";
   srand(time(NULL));
   uint64_t ret;
   uint8_t *oprs = (uint8_t *)malloc(ITR*sizeof(uint8_t));
   uint64_t *opr_vals = (uint64_t *)malloc(ITR*sizeof(uint64_t));

   for (uint64_t i = 0; i < ITR; i++) {
      oprs[i] = rand() % 3;
      if (oprs[i] == 0) { // delete
         opr_vals[i] = vals[rand() % nvals];
      } else if (oprs[i] == 1) { // query
         opr_vals[i] = vals[rand() % nvals];
      } else if (oprs[i] == 2) { // insert
         opr_vals[i] = other_vals[rand() % nvals];
      }
   }
   gettimeofday(&start, &tzp);
   for (uint64_t i = 0; i < ITR; i++) {
      if (oprs[i] == 0) { // delete
         ret = vqf_remove(filter, opr_vals[i]);
      } else if (oprs[i] == 1) { // query
         ret = vqf_is_present(filter, opr_vals[i]);
      } else if (oprs[i] == 2) { // insert
         if (!vqf_insert(filter, opr_vals[i])) {
            fprintf(stderr, "Insertion failed");
            exit(EXIT_FAILURE);
         }
      }
   }
   gettimeofday(&end, &tzp);
   std::cout << "ret: " << ret << '\n';
   print_time_elapsed("Workload time", &start, &end, ITR, "operations");

   return 0;
}
