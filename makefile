# *********************************************
#	Makefile pour la library _sspropvc.so
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


CLEANFILES = *.o *.a *.so


_sspropvc.so:		sspropvc_plain.o sspropvc_wrap.o libssprop.a
					icc -xT -shared  sspropvc_plain.o sspropvc_wrap.o -o _sspropvc.so -lfftw3 -lm -lpython2.5

sspropvc_wrap.o:	sspropvc_wrap.c
					icc -c -O3 -fomit-frame-pointer -fPIC -xT -m64 sspropvc_wrap.c -I/usr/include/python2.5

sspropsc_wrap.o:	sspropsc_wrap.c
					icc -c -O3 -fomit-frame-pointer -fPIC -xT -m64 sspropsc_wrap.c -I/usr/include/python2.5

sspropvc_plain.o:	sspropvc_plain.c
					icc -o sspropvc_plain.o -c -O3 -fPIC -xT -mtune=core2 -m64 sspropvc_plain.c

sspropvc_wrap.c:	sspropvc.i sspropvc.h numpy.i
					swig -python sspropvc.i

sspropsc_wrap.c:	sspropsc.i sspropsc.h numpy.i
					swig -python sspropsc.i

libssprop.a:		sspropvc_plain.o
					ar rc libssprop.a sspropvc_plain.o


# *********************************************
# Clean-up rules
# *********************************************

clean:
	-rm -f $(CLEANFILES)

					

