TARGETS= main main_tx main_id bm

OPT=-Ofast -g

ARCH=-msse4.2

ifeq ($(P),1)
   OPT=-g -no-pie
endif

HAVE_AVX512=$(filter-out 0,$(shell lscpu | grep avx512bw | wc -l))

ifeq ($(THREAD),1)
   OPT +=-DENABLE_THREADS
endif

CXX = g++ -std=c++11 -fgnu-tm -frename-registers  -march=native
CC = gcc -std=gnu11 -fgnu-tm -frename-registers  -march=native
LD= g++ -std=c++11

LOC_INCLUDE=include
LOC_SRC=src
OBJDIR=obj

CXXFLAGS += -Wall $(DEBUG) $(PROFILE) $(OPT) $(ARCH) -m64 -I. -I$(LOC_INCLUDE)

CFLAGS += -Wall $(DEBUG) $(PROFILE) $(OPT) $(ARCH) -m64 -I. -I$(LOC_INCLUDE)

LDFLAGS += $(DEBUG) $(PROFILE) $(OPT) -lpthread -lssl -lcrypto -lm -litm

#
# declaration of dependencies
#

all: $(TARGETS)

# dependencies between programs and .o files
ifeq ($(HAVE_AVX512),1)
main:							$(OBJDIR)/main.o $(OBJDIR)/vqf_filter.o $(OBJDIR)/shuffle_matrix_512.o $(OBJDIR)/shuffle_matrix_512_16.o 
main_id:						$(OBJDIR)/main_id.o $(OBJDIR)/vqf_filter.o $(OBJDIR)/shuffle_matrix_512.o $(OBJDIR)/shuffle_matrix_512_16.o 
main_tx:						$(OBJDIR)/main_tx.o $(OBJDIR)/vqf_filter.o $(OBJDIR)/shuffle_matrix_512.o $(OBJDIR)/shuffle_matrix_512_16.o 
bm:							$(OBJDIR)/bm.o $(OBJDIR)/vqf_filter.o $(OBJDIR)/shuffle_matrix_512.o $(OBJDIR)/shuffle_matrix_512_16.o 
else
main:							$(OBJDIR)/main.o $(OBJDIR)/vqf_filter.o 
main_id:						$(OBJDIR)/main_id.o $(OBJDIR)/vqf_filter.o
main_tx:						$(OBJDIR)/main_tx.o $(OBJDIR)/vqf_filter.o
bm:							$(OBJDIR)/bm.o $(OBJDIR)/vqf_filter.o 
endif

# dependencies between .o files and .cc (or .c) files
$(OBJDIR)/shuffle_matrix_512_16.o: 	$(LOC_SRC)/shuffle_matrix_512_16.c
$(OBJDIR)/shuffle_matrix_512.o: 	$(LOC_SRC)/shuffle_matrix_512.c
$(OBJDIR)/main.o: 			$(LOC_SRC)/main.cc
$(OBJDIR)/main_id.o: 			$(LOC_SRC)/main_id.cc
$(OBJDIR)/main_tx.o: 			$(LOC_SRC)/main_tx.cc
$(OBJDIR)/bm.o: 			$(LOC_SRC)/bm.cc

$(OBJDIR)/vqf_filter.o: 			$(LOC_SRC)/vqf_filter.c

#
# generic build rules
#

$(TARGETS):
	$(LD) $^ $(LDFLAGS) -o $@

$(OBJDIR)/%.o: $(LOC_SRC)/%.cc | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c -o $@ $<

$(OBJDIR)/%.o: $(LOC_SRC)/%.c | $(OBJDIR)
	$(CXX) $(CFLAGS) $(INCLUDE) -c -o $@ $<

$(OBJDIR):
	@mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR) core $(TARGETS)

