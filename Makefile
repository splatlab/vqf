TARGETS= main main_tx main_id bm

ifdef D
	DEBUG=-g
	OPT=
else
	DEBUG=
	OPT=-Ofast
endif

ifdef NH
	ARCH=
else
	ARCH=-msse4.2
endif

ifdef P
	PROFILE=-pg -no-pie # for bug in gprof.
endif

CXX = g++ -std=c++11 -fgnu-tm -mavx512bw -mavx512f -frename-registers  -march=native
CC = gcc -std=gnu11 -fgnu-tm -mavx512bw -mavx512f -frename-registers  -march=native
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
main:							$(OBJDIR)/main.o $(OBJDIR)/ququ_filter.o $(OBJDIR)/shuffle_matrix_512.o $(OBJDIR)/shuffle_matrix_512_16.o
main_id:						$(OBJDIR)/main_id.o $(OBJDIR)/ququ_filter.o $(OBJDIR)/shuffle_matrix_512.o $(OBJDIR)/shuffle_matrix_512_16.o
bm:							$(OBJDIR)/bm.o $(OBJDIR)/ququ_filter.o $(OBJDIR)/shuffle_matrix_512.o $(OBJDIR)/shuffle_matrix_512_16.o

# dependencies between .o files and .cc (or .c) files
$(OBJDIR)/shuffle_matrix_512_16.o: 	$(LOC_SRC)/shuffle_matrix_512_16.c
$(OBJDIR)/shuffle_matrix_512.o: 	$(LOC_SRC)/shuffle_matrix_512.c
$(OBJDIR)/main.o: 			$(LOC_SRC)/main.cc
$(OBJDIR)/main_id.o: 			$(LOC_SRC)/main_id.cc
$(OBJDIR)/bm.o: 			$(LOC_SRC)/bm.cc

$(OBJDIR)/ququ_filter.o: 			$(LOC_SRC)/ququ_filter.c

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

