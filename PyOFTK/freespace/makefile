# *********************************************
#	Makefile for the _fkabcd.so library
#
#
#
#
#
#	Author: 	Martin Laprise
#		    	Universite Laval
#				martin.laprise.1@ulaval.ca
#                 
# *********************************************


CLEANFILES = *.o *.a *.so *_wrap*

LBITS := $(shell getconf LONG_BIT)
ifeq ($(LBITS),64)

_fkabcd.so:			fkabcd.o fkabcd_wrap.o libfkabcd.a
					g++ -shared fkabcd.o fkabcd_wrap.o -o _fkabcd.so -lpython2.6 -lfftw -lm

libfkabcd.a:		fkabcd.o
					ar rc libfkabcd.a fkabcd.o

fkabcd_wrap.cpp:	fkabcd.i fkabcd.h numpy.i
					swig -python -c++ fkabcd.i

fkabcd_wrap.o:		fkabcd_wrap.cpp
					gcc -o fkabcd_wrap.o -c -O3 -fomit-frame-pointer -fPIC -funroll-loops -m64 fkabcd_wrap.cxx -I/usr/local/lib/python2.6/dist-packages/numpy/core/include/ -I/usr/include/python2.6/

fkabcd.o:			fkabcd.cpp
					gcc -o fkabcd.o -c -O3 -fomit-frame-pointer -fPIC -funroll-loops -m64 fkabcd.cpp
else

_fkabcd.so:			fkabcd.o fkabcd_wrap.o libfkabcd.a
					g++ -shared fkabcd.o fkabcd_wrap.o -o _fkabcd.so -lpython2.6 -lfftw -lm

libfkabcd.a:		fkabcd.o
					ar rc libfkabcd.a fkabcd.o

fkabcd_wrap.cpp:	fkabcd.i fkabcd.h numpy.i
					swig -python -c++ fkabcd.i

fkabcd_wrap.o:		fkabcd_wrap.cpp
					gcc -o fkabcd_wrap.o -c -O3 -fomit-frame-pointer -fPIC -funroll-loops fkabcd_wrap.cxx -I/usr/local/lib/python2.6/dist-packages/

fkabcd.o:			fkabcd.cpp
					gcc -o fkabcd.o -c -O3 -fomit-frame-pointer -fPIC -funroll-loops fkabcd.cpp

endif
# *********************************************
# Clean-up rules
# *********************************************

clean:
	-rm -f $(CLEANFILES)

					

