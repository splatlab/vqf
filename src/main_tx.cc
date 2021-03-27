/*
 * ============================================================================
 *
 *        Authors:  Prashant Pandey <ppandey@cs.stonybrook.edu>
 *                  Rob Johnson <robj@vmware.com>   
 *
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h>
#include <openssl/rand.h>

#include "vqf_filter.h"

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


typedef struct args {
   vqf_filter *cf;
   uint64_t *vals;
   uint64_t start;
   uint64_t end;
} args;

void *insert_bm(void *arg)
{
   args *a = (args *)arg;
   for (uint32_t i = a->start; i <= a->end; i++) {
      int ret = vqf_insert(a->cf, a->vals[i]);
      if (ret < 0) {
         fprintf(stderr, "failed insertion for key: %lx.\n", a->vals[i]);
         abort();
      }
   }
   return NULL;
}

void *query_bm(void *arg)
{
   args *a = (args *)arg;
   for (uint32_t i = a->start; i <= a->end; i++) {
      int ret = vqf_is_present(a->cf, a->vals[i]);
      if (ret < 0) {
         fprintf(stderr, "failed insertion for key: %lx.\n", a->vals[i]);
         abort();
      }
   }
   return NULL;
}

void multi_threaded_insertion(args args[], int tcnt)
{
   pthread_t threads[tcnt];

   for (int i = 0; i < tcnt; i++) {
      fprintf(stdout, "Thread %d bounds %ld %ld\n", i, args[i].start, args[i].end);
      if (pthread_create(&threads[i], NULL, &insert_bm, &args[i])) {
         fprintf(stderr, "Error creating thread\n");
         exit(0);
      }
   }

   for (int i = 0; i < tcnt; i++) {
      if (pthread_join(threads[i], NULL)) {
         fprintf(stderr, "Error joining thread\n");
         exit(0);
      }
   }
}

int main(int argc, char **argv)
{
   if (argc < 3) {
      fprintf(stderr, "Please specify three arguments: \n \
            1. log of the number of slots in the CQF.\n \
            2. number of threads.\n");
      exit(1);
   }
   uint64_t qbits = atoi(argv[1]);
   uint32_t tcnt = atoi(argv[2]);
   uint64_t nhashbits = qbits + 8;
   uint64_t nslots = (1ULL << qbits);
   uint64_t nvals = 85*nslots/100;

   uint64_t *vals;
   vqf_filter *filter;	

   /* initialize vqf filter */
   if ((filter = vqf_init(nslots)) == NULL) {
      fprintf(stderr, "Can't allocate vqf filter.");
      exit(EXIT_FAILURE);
   }

   /* Generate random values */
   vals = (uint64_t*)calloc(nvals, sizeof(vals[0]));
   RAND_bytes((unsigned char *)vals, sizeof(*vals) * nvals);
   for (uint32_t i = 0; i < nvals; i++) {
      vals[i] = (1 * vals[i]) % filter->metadata.range;
   }

   args *arg = (args*)malloc(tcnt * sizeof(args));
   for (uint32_t i = 0; i < tcnt; i++) {
      arg[i].cf = filter;
      arg[i].vals = vals;
      arg[i].start = (nvals/tcnt) * i;
      arg[i].end = (nvals/tcnt) * (i + 1) - 1;
   }
   //fprintf(stdout, "Total number of items: %ld\n", arg[tcnt-1].end);

   struct timeval start, end;
   struct timezone tzp;

   gettimeofday(&start, &tzp);
   multi_threaded_insertion(arg, tcnt);
   gettimeofday(&end, &tzp);
   print_time_elapsed("Insertion time", &start, &end, nvals, "insert");

   //fprintf(stdout, "Inserted all items: %ld\n", arg[tcnt-1].end);

   for (uint64_t i = 0; i < arg[tcnt-1].end; i++) {
      if (!vqf_is_present(filter, vals[i])) {
         fprintf(stderr, "Lookup failed for %ld", vals[i]);
         exit(EXIT_FAILURE);
      }
   }

   return 0;
}
