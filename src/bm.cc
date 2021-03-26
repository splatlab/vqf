/*
 * =====================================================================================
 *
 *       Filename:  bm.c
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  05/18/2015 08:54:53 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Prashant Pandey (ppandey@cs.stonybrook.edu),
 *   Organization:
 *
 * =====================================================================================
 */

#include <assert.h>
#include <openssl/rand.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "vqf_wrapper.h"

typedef void *(*rand_init)(uint64_t maxoutputs, __uint128_t maxvalue,
                           void *params);
typedef int (*gen_rand)(void *state, uint64_t noutputs, __uint128_t *outputs);
typedef void *(*duplicate_rand)(void *state);

typedef int (*init_op)(uint64_t nvals);
typedef int (*insert_op)(__uint128_t val);
typedef int (*lookup_op)(__uint128_t val);
typedef int (*remove_op)(__uint128_t val);
typedef __uint128_t (*get_range_op)();
typedef int (*destroy_op)();

typedef struct rand_generator {
  rand_init init;
  gen_rand gen;
  duplicate_rand dup;
} rand_generator;

typedef struct filter {
  init_op init;
  insert_op insert;
  lookup_op lookup;
	remove_op remove;
  get_range_op range;
  destroy_op destroy;
} filter;

typedef struct uniform_pregen_state {
  uint64_t maxoutputs;
  uint64_t nextoutput;
  __uint128_t *outputs;
} uniform_pregen_state;

typedef struct uniform_online_state {
  uint64_t maxoutputs;
  __uint128_t maxvalue;
  unsigned int seed;
  char *buf;
  int STATELEN;
  struct random_data *rand_state;
} uniform_online_state;

void *uniform_pregen_init(uint64_t maxoutputs, __uint128_t maxvalue,
                          void *params) {
  uint32_t i;
  uniform_pregen_state *state =
      (uniform_pregen_state *)malloc(sizeof(uniform_pregen_state));
  assert(state != NULL);

  state->nextoutput = 0;

  state->maxoutputs = maxoutputs;
  state->outputs =
      (__uint128_t *)malloc(state->maxoutputs * sizeof(state->outputs[0]));
  assert(state->outputs != NULL);
  uint64_t nbytes = sizeof(*state->outputs) * state->maxoutputs;
  uint8_t *ptr = (unsigned char *)state->outputs;
	while (nbytes > (1ULL << 30)) {
		RAND_bytes(ptr, 1ULL << 30);
		ptr += (1ULL << 30);
		nbytes -= (1ULL << 30);
	}
	RAND_bytes(ptr, nbytes);
  for (i = 0; i < state->maxoutputs; i++)
    state->outputs[i] = (1 * state->outputs[i]) % maxvalue;

  return (void *)state;
}

int uniform_pregen_gen_rand(void *_state, uint64_t noutputs,
                            __uint128_t *outputs) {
  uniform_pregen_state *state = (uniform_pregen_state *)_state;
  assert(state->nextoutput + noutputs <= state->maxoutputs);
  memcpy(outputs, state->outputs + state->nextoutput,
         noutputs * sizeof(*state->outputs));
  state->nextoutput += noutputs;
  return noutputs;
}

void *uniform_pregen_duplicate(void *state) {
  uniform_pregen_state *newstate =
      (uniform_pregen_state *)malloc(sizeof(*newstate));
  assert(newstate);
  memcpy(newstate, state, sizeof(*newstate));
  return newstate;
}

void *uniform_online_init(uint64_t maxoutputs, __uint128_t maxvalue,
                          void *params) {
  uniform_online_state *state =
      (uniform_online_state *)malloc(sizeof(uniform_online_state));
  assert(state != NULL);

  state->maxoutputs = maxoutputs;
  state->maxvalue = maxvalue;
  state->seed = time(NULL);
  state->STATELEN = 256;
  state->buf = (char *)calloc(256, sizeof(char));
  state->rand_state =
      (struct random_data *)calloc(1, sizeof(struct random_data));

  initstate_r(state->seed, state->buf, state->STATELEN, state->rand_state);
  return (void *)state;
}

int uniform_online_gen_rand(void *_state, uint64_t noutputs,
                            __uint128_t *outputs) {
  uint32_t i, j;
  uniform_online_state *state = (uniform_online_state *)_state;
  assert(state->rand_state != NULL);
  memset(outputs, 0, noutputs * sizeof(__uint128_t));
  for (i = 0; i < noutputs; i++) {
    int32_t result;
    for (j = 0; j < 4; j++) {
      random_r(state->rand_state, &result);
      outputs[i] = (outputs[i] * RAND_MAX) + result;
    }
    outputs[i] = (1 * outputs[i]) % state->maxvalue;
  }
  return noutputs;
}

void *uniform_online_duplicate(void *_state) {
  uniform_online_state *newstate =
      (uniform_online_state *)malloc(sizeof(uniform_online_state));
  assert(newstate != NULL);
  uniform_online_state *oldstate = (uniform_online_state *)_state;

  newstate->maxvalue = oldstate->maxvalue;
  newstate->seed = oldstate->seed;
  newstate->STATELEN = oldstate->STATELEN;

  newstate->buf = (char *)calloc(256, sizeof(char));
  memcpy(newstate->buf, oldstate->buf, newstate->STATELEN);
  newstate->rand_state =
      (struct random_data *)calloc(1, sizeof(struct random_data));

  initstate_r(newstate->seed, newstate->buf, newstate->STATELEN,
              newstate->rand_state);
  return newstate;
}

rand_generator uniform_pregen = {uniform_pregen_init, uniform_pregen_gen_rand,
                                 uniform_pregen_duplicate};

rand_generator uniform_online = {uniform_online_init, uniform_online_gen_rand,
                                 uniform_online_duplicate};

filter cf = {q_init, q_insert, q_lookup, q_remove, q_range, q_destroy};

uint64_t tv2usec(struct timeval tv) {
  return 1000000 * tv.tv_sec + tv.tv_usec;
}

uint64_t tv2msec(struct timeval tv) {
  return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

int cmp_uint64_t(const void *a, const void *b) {
  const uint64_t *ua = (const uint64_t *)a, *ub = (const uint64_t *)b;
  return *ua < *ub ? -1 : *ua == *ub ? 0 : 1;
}

void usage(char *name) {
  printf(
      "%s [OPTIONS]\n"
      "Options are:\n"
      "  -n nslots     [ log_2 of filter capacity.  Default 24 ]\n"
      "  -r nruns      [ number of runs.  Default 1 ]\n"
      "  -p npoints    [ number of points on the graph.  Default 20 ]\n"
      "  -m randmode   [ Data distribution, one of \n"
      "                    uniform_pregen\n"
      "                    uniform_online\n"
      "                    zipfian_pregen\n"
      "                  Default uniform_pregen ]\n"
      "  -d datastruct  [ Default qf. ]\n"
      "  -f outputfile  [ Default qf. ]\n",
      name);
}

int main(int argc, char **argv) {
  uint32_t nbits = 0, nruns = 0;
  unsigned int npoints = 0;
  uint64_t nslots = 0, nvals = 0;
  char *randmode = "uniform_pregen";
  char *datastruct = "qf";
  char *outputfile = "qf";

  filter filter_ds;
  rand_generator *vals_gen;
  void *vals_gen_state;
  void *old_vals_gen_state;
  void *remove_vals_gen_state;
  rand_generator *othervals_gen;
  void *othervals_gen_state;

  //	__uint128_t *vals;
  //	__uint128_t *othervals;

  unsigned int i, j, exp, run;
  struct timeval tv_insert[100][1];
  struct timeval tv_exit_lookup[100][1];
  struct timeval tv_false_lookup[100][1];
  struct timeval tv_remove[100][1];
  uint64_t fps = 0;

  FILE *fp_insert;
  FILE *fp_exit_lookup;
  FILE *fp_false_lookup;
  FILE *fp_remove;
  const char *dir = "./";
  const char *insert_op = "-insert.txt\0";
  const char *exit_lookup_op = "-exists-lookup.txt\0";
  const char *false_lookup_op = "-false-lookup.txt\0";
  const char *remove_op = "-remove.txt\0";
  char filename_insert[256];
  char filename_exit_lookup[256];
  char filename_false_lookup[256];
  char filename_remove[256];

  /* Argument parsing */
  int opt;
  char *term;

  while ((opt = getopt(argc, argv, "n:r:p:m:d:f:")) != -1) {
    switch (opt) {
      case 'n':
        nbits = strtol(optarg, &term, 10);
        if (*term) {
          fprintf(stderr, "Argument to -n must be an integer\n");
          usage(argv[0]);
          exit(1);
        }
        nslots = (1ULL << nbits);
        nvals = 950 * nslots / 1000;
        break;
      case 'r':
        nruns = strtol(optarg, &term, 10);
        if (*term) {
          fprintf(stderr, "Argument to -r must be an integer\n");
          usage(argv[0]);
          exit(1);
        }
        break;
      case 'p':
        npoints = strtol(optarg, &term, 10);
        if (*term) {
          fprintf(stderr, "Argument to -p must be an integer\n");
          usage(argv[0]);
          exit(1);
        }
        break;
      case 'm':
        randmode = optarg;
        break;
      case 'd':
        datastruct = optarg;
        break;
      case 'f':
        outputfile = optarg;
        break;
      default:
        fprintf(stderr, "Unknown option\n");
        usage(argv[0]);
        exit(1);
        break;
    }
  }

  if (strcmp(randmode, "uniform_pregen") == 0) {
    vals_gen = &uniform_pregen;
    othervals_gen = &uniform_pregen;
  } else if (strcmp(randmode, "uniform_online") == 0) {
    vals_gen = &uniform_online;
    othervals_gen = &uniform_online;
  } else {
    fprintf(stderr, "Unknown randmode.\n");
    usage(argv[0]);
    exit(1);
  }

  if (strcmp(datastruct, "cf") == 0) {
    filter_ds = cf;
    //	} else if (strcmp(datastruct, "gqf") == 0) {
    //		filter_ds = gqf;
    //	} else if (strcmp(datastruct, "qf") == 0) {
    //		filter_ds = qf;
    //	} else if (strcmp(datastruct, "bf") == 0) {
    //		filter_ds = bf;
  } else {
    fprintf(stderr, "Unknown randmode.\n");
    usage(argv[0]);
    exit(1);
  }

  snprintf(filename_insert,
           strlen(dir) + strlen(outputfile) + strlen(insert_op) + 1, "%s%s%s",
           dir, outputfile, insert_op);
  snprintf(filename_exit_lookup,
           strlen(dir) + strlen(outputfile) + strlen(exit_lookup_op) + 1,
           "%s%s%s", dir, outputfile, exit_lookup_op);

  snprintf(filename_false_lookup,
           strlen(dir) + strlen(outputfile) + strlen(false_lookup_op) + 1,
           "%s%s%s", dir, outputfile, false_lookup_op);
  snprintf(filename_remove,
           strlen(dir) + strlen(outputfile) + strlen(remove_op) + 1, "%s%s%s",
           dir, outputfile, remove_op);

  fp_insert = fopen(filename_insert, "w");
  fp_exit_lookup = fopen(filename_exit_lookup, "w");
  fp_false_lookup = fopen(filename_false_lookup, "w");
  fp_remove = fopen(filename_remove, "w");

	if (fp_insert == NULL || fp_exit_lookup == NULL || fp_false_lookup == NULL
			|| fp_remove == NULL) {
    printf("Can't open the data file");
    exit(1);
  }

  fprintf(fp_insert, "x_0");
  for (run = 0; run < nruns; run++) {
    fprintf(fp_insert, "    y_%d", run);
  }
  fprintf(fp_insert, "\n");

  fprintf(fp_exit_lookup, "x_0");
  for (run = 0; run < nruns; run++) {
    fprintf(fp_exit_lookup, "    y_%d", run);
  }
  fprintf(fp_exit_lookup, "\n");

  fprintf(fp_false_lookup, "x_0");
  for (run = 0; run < nruns; run++) {
    fprintf(fp_false_lookup, "    y_%d", run);
  }
  fprintf(fp_false_lookup, "\n");

	fprintf(fp_remove, "x_0");
  for (run = 0; run < nruns; run++) {
    fprintf(fp_remove, "    y_%d", run);
  }
  fprintf(fp_remove, "\n");

  fclose(fp_insert);
  fclose(fp_exit_lookup);
  fclose(fp_false_lookup);
  fclose(fp_remove);

  for (run = 0; run < nruns; run++) {
    fps = 0;
    filter_ds.init(nbits);

    vals_gen_state = vals_gen->init(nvals, filter_ds.range(), NULL);
    old_vals_gen_state = vals_gen->dup(vals_gen_state);
    remove_vals_gen_state = vals_gen->dup(vals_gen_state);
    sleep(5);
    othervals_gen_state = othervals_gen->init(nvals, filter_ds.range(), NULL);

    for (exp = 0; exp < 2 * npoints; exp += 2) {
      fp_insert = fopen(filename_insert, "a");
      fp_exit_lookup = fopen(filename_exit_lookup, "a");
      fp_false_lookup = fopen(filename_false_lookup, "a");

      i = (exp / 2) * (nvals / npoints);
      j = ((exp / 2) + 1) * (nvals / npoints);
      printf("Round: %d\n", exp / 2);

      gettimeofday(&tv_insert[exp][run], NULL);
      for (; i < j; i += 1 << 16) {
        int nitems = j - i < 1 << 16 ? j - i : 1 << 16;
        __uint128_t vals[1 << 16];
        int m;
        assert(vals_gen->gen(vals_gen_state, nitems, vals) == nitems);

        for (m = 0; m < nitems; m++) {
          filter_ds.insert(vals[m]);
        }
      }
      gettimeofday(&tv_insert[exp + 1][run], NULL);
      fprintf(fp_insert, "%d", ((exp / 2) * (100 / npoints)));
      fprintf(fp_insert, " %f\n",
               1.0 * (nvals / npoints) /
                  (tv2usec(tv_insert[exp + 1][run]) -
                   tv2usec(tv_insert[exp][run])));

      i = (exp / 2) * (nvals / 20);
      gettimeofday(&tv_exit_lookup[exp][run], NULL);
      for (; i < j; i += 1 << 16) {
        int nitems = j - i < 1 << 16 ? j - i : 1 << 16;
        __uint128_t vals[1 << 16];
        int m;
        assert(vals_gen->gen(old_vals_gen_state, nitems, vals) == nitems);
        for (m = 0; m < nitems; m++) {
          if (!filter_ds.lookup(vals[m])) {
            // fprintf(stderr, "Failed lookup for 0x%lx%016lx\n",
            //(uint64_t)(vals[m] >> 64),
            //(uint64_t)(vals[m] & 0xffffffffffffffff));
            // abort();
          }
        }
      }
      gettimeofday(&tv_exit_lookup[exp + 1][run], NULL);
      fprintf(fp_exit_lookup, "%d", ((exp / 2) * (100 / npoints)));
      fprintf(fp_exit_lookup, " %f\n",
              1.0 * (nvals / npoints) /
                  (tv2usec(tv_exit_lookup[exp + 1][run]) -
                   tv2usec(tv_exit_lookup[exp][run])));

      i = (exp / 2) * (nvals / 20);
      gettimeofday(&tv_false_lookup[exp][run], NULL);
      for (; i < j; i += 1 << 16) {
        int nitems = j - i < 1 << 16 ? j - i : 1 << 16;
        __uint128_t othervals[1 << 16];
        int m;
        assert(othervals_gen->gen(othervals_gen_state, nitems, othervals) ==
               nitems);
        for (m = 0; m < nitems; m++) {
          fps += filter_ds.lookup(othervals[m]);
        }
      }
      gettimeofday(&tv_false_lookup[exp + 1][run], NULL);
      fprintf(fp_false_lookup, "%d", ((exp / 2) * (100 / npoints)));
      fprintf(fp_false_lookup, " %f\n",
              1.0 * (nvals / npoints) /
                  (tv2usec(tv_false_lookup[exp + 1][run]) -
                   tv2usec(tv_false_lookup[exp][run])));

      fclose(fp_insert);
      fclose(fp_exit_lookup);
      fclose(fp_false_lookup);
    }

    for (exp = 0; exp < 2 * npoints; exp += 2) {
       fp_remove = fopen(filename_remove, "a");
       i = (exp / 2) * (nvals / npoints);
       j = ((exp / 2) + 1) * (nvals / npoints);
       printf("Round: %d\n", exp / 2);

       gettimeofday(&tv_remove[exp][run], NULL);
       for (; i < j; i += 1 << 16) {
          int nitems = j - i < 1 << 16 ? j - i : 1 << 16;
          __uint128_t vals[1 << 16];
          int m;
          assert(vals_gen->gen(remove_vals_gen_state, nitems, vals) == nitems);

          for (m = 0; m < nitems; m++) {
             filter_ds.remove(vals[m]);
          }
       }
       gettimeofday(&tv_remove[exp + 1][run], NULL);
       fprintf(fp_remove, "%d", ((exp / 2) * (100 / npoints)));
       fprintf(fp_remove, " %f\n",
             1.0 * (nvals / npoints) /
             (tv2usec(tv_remove[exp + 1][run]) -
              tv2usec(tv_remove[exp][run])));

       fclose(fp_remove);
   }

    filter_ds.destroy();
  }
  printf("Insert Performance written to file: %s\n", filename_insert);
  printf("Exist lookup Performance written to file: %s\n", filename_exit_lookup);
  printf("False lookup Performance written to file: %s\n", filename_false_lookup);
  printf("Remove Performance written to file: %s\n", filename_remove);

  printf("FP rate: %f (%lu/%lu)\n", 1.0 * fps / nvals, fps, nvals);

  return 0;
}
