# *********************************************
#	Makefile for the ssprop library
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

all:				_sspropvc.so _sspropsc.so

_sspropsc.so:		sspropsc_plain.o sspropsc_wrap.o libssprop.a
					g++ -shared  sspropsc_plain.o sspropsc_wrap.o -o _sspropsc.so -lfftw3 -lm -lpython2.6

_sspropvc.so:		sspropvc_plain.o sspropvc_wrap.o libssprop.a
					g++ -shared  sspropvc_plain.o sspropvc_wrap.o -o _sspropvc.so -lfftw3 -lm -lpython2.6

sspropvc_wrap.o:	sspropvc_wrap.c
					g++ -c -m64 -O3 -fomit-frame-pointer -fPIC -funroll-loops -fstrict-aliasing sspropvc_wrap.c -I/usr/include/python2.6 -I/usr/local/lib/python2.6/dist-packages/numpy/core/include/

sspropsc_wrap.o:	sspropsc_wrap.c
					g++ -c -m64 -O3 -fomit-frame-pointer -fPIC -funroll-loops -fstrict-aliasing sspropsc_wrap.c -I/usr/include/python2.6 -I/usr/local/lib/python2.6/dist-packages/numpy/core/include/

sspropvc_plain.o:	sspropvc_plain.c
					g++ -c -m64 -O3 -fomit-frame-pointer -fPIC -funroll-loops -fstrict-aliasing sspropvc_plain.c

sspropsc_plain.o:	sspropsc_plain.c
					g++ -c -m64 -O3 -fomit-frame-pointer -fPIC -funroll-loops -fstrict-aliasing sspropsc_plain.c

sspropvc_wrap.c:	sspropvc.i sspropvc.h numpy.i
					swig -python sspropvc.i

sspropsc_wrap.c:	sspropsc.i sspropsc.h numpy.i
					swig -python sspropsc.i

libssprop.a:		sspropvc_plain.o sspropsc_plain.o
					ar rc libssprop.a sspropvc_plain.o

else

all:				_sspropvc.so _sspropsc.so

_sspropsc.so:		sspropsc_plain.o sspropsc_wrap.o libssprop.a
					g++ -shared  sspropsc_plain.o sspropsc_wrap.o -o _sspropsc.so -lfftw3 -lm -lpython2.6

_sspropvc.so:		sspropvc_plain.o sspropvc_wrap.o libssprop.a
					g++ -shared  sspropvc_plain.o sspropvc_wrap.o -o _sspropvc.so -lfftw3 -lm -lpython2.6

sspropvc_wrap.o:	sspropvc_wrap.c
					g++ -c -O3 -fomit-frame-pointer -fPIC -funroll-loops -fstrict-aliasing sspropvc_wrap.c -I/usr/include/python2.6

sspropsc_wrap.o:	sspropsc_wrap.c
					g++ -c -O3 -fomit-frame-pointer -fPIC -funroll-loops -fstrict-aliasing sspropsc_wrap.c -I/usr/include/python2.6

sspropvc_plain.o:	sspropvc_plain.c
					g++ -c -O3 -fomit-frame-pointer -fPIC -funroll-loops -fstrict-aliasing sspropvc_plain.c

sspropsc_plain.o:	sspropsc_plain.c
					g++ -c -O3 -fomit-frame-pointer -fPIC -funroll-loops -fstrict-aliasing sspropsc_plain.c

sspropvc_wrap.c:	sspropvc.i sspropvc.h numpy.i
					swig -python sspropvc.i

sspropsc_wrap.c:	sspropsc.i sspropsc.h numpy.i
					swig -python sspropsc.i

libssprop.a:		sspropvc_plain.o sspropsc_plain.o
					ar rc libssprop.a sspropvc_plain.o

endif



# *********************************************
# Clean-up rules
# *********************************************

clean:
	-rm -f $(CLEANFILES)

					

