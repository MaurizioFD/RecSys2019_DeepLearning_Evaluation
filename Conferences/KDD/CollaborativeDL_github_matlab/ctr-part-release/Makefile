CC = g++ -Wall
#CC = g++ -ansi -Wall -pedantic
#CFLAGS = -g -Wall -O3 -ffast-math -DHAVE_INLINE -DGSL_RANGE_CHECK_OFF
# CFLAGS = -g -Wall
LDFLAGS = -lgsl -lm -lgslcblas

GSL_INCLUDE = /usr/local/gsl/include
GSL_LIB = /usr/local/gsl/lib

LSOURCE = main.cpp utils.cpp ctr.cpp data.cpp
LHEADER = utils.h ctr.h data.h

ctr: $(LSOURCE) $(HEADER)
	  $(CC) -I$(GSL_INCLUDE) -L$(GSL_LIB) $(LSOURCE) -o $@ $(LDFLAGS)

ctr-d: $(LSOURCE) $(HEADER)
	  $(CC) -g -I$(GSL_INCLUDE) -L$(GSL_LIB) $(LSOURCE) -o $@ $(LDFLAGS)

clean:
	-rm -f *.o ctr
clean-d:
	-rm -f *.o ctr-d
