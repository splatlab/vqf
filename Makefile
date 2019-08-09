TARGETS= main bm

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
	ARCH=-msse4.2 -D__SSE4_2_
endif

ifdef P
	PROFILE=-pg -no-pie # for bug in gprof.
endif

CXX = g++ -std=c++11 -mavx2
CC = gcc -std=gnu11 -mavx2
LD= g++ -std=c++11

LOC_INCLUDE=include
LOC_SRC=src
OBJDIR=obj

CXXFLAGS += -Wall $(DEBUG) $(PROFILE) $(OPT) $(ARCH) -m64 -I. -I$(LOC_INCLUDE)

CFLAGS += -Wall $(DEBUG) $(PROFILE) $(OPT) $(ARCH) -m64 -I. -I$(LOC_INCLUDE)

LDFLAGS += $(DEBUG) $(PROFILE) $(OPT) -lpthread -lssl -lcrypto -lm

#
# declaration of dependencies
#

all: $(TARGETS)

# dependencies between programs and .o files
main:							$(OBJDIR)/main.o $(OBJDIR)/ququ_filter.o
bm:							$(OBJDIR)/bm.o $(OBJDIR)/ququ_filter.o

# dependencies between .o files and .cc (or .c) files
$(OBJDIR)/main.o: 			$(LOC_SRC)/main.cc
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

