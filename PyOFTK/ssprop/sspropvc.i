/* *********************************************
#	SWIG Interface to sspropvc_plain.c 
#
#
#
#
#	Author: 	Martin Laprise
#		    	Universite Laval
#				martin.laprise.1@ulaval.ca
#                 
# ********************************************* */


%module sspropvc

%{
	#define SWIG_FILE_WITH_INIT
	#define REAL double
	#define COMPLEX fftw_complex
%}

%{
	#include<math.h>
	#include"sspropvc.h"
	#include <numpy/arrayobject.h>
%}

%include "carrays.i"
%include "numpy.i"

%init
%{
	import_array();
%}

%numpy_typemaps(Py_complex , NPY_CDOUBLE, int)
%numpy_typemaps(fftwf_complex , NPY_CFLOAT , int)
%numpy_typemaps(fftw_complex, NPY_CDOUBLE, int)

%array_class(double, doubleArray);

%apply (fftw_complex* IN_ARRAY1, int DIM1) {(fftw_complex* u0x, int nu0x)};
%apply (fftw_complex* IN_ARRAY1, int DIM1) {(fftw_complex* u0y, int nu0y)};
%apply (fftw_complex* ARGOUT_ARRAY1, int DIM1) {(fftw_complex *u1x, int nu1x)};
%apply (fftw_complex* ARGOUT_ARRAY1, int DIM1) {(fftw_complex *u1y, int nu1y)};
%apply (double ARGOUT_ARRAY1[ANY]) {(double usefulStuff[10])};
%apply (int DIM1, double* IN_ARRAY1) {(int nalphaa, double* alphaa)};
%apply (int DIM1, double* IN_ARRAY1) {(int nalphab, double* alphab)};
%apply (int DIM1, double* IN_ARRAY1) {(int nbetaa, double* betaa)};
%apply (int DIM1, double* IN_ARRAY1) {(int nbetab, double* betab)};

extern void vector(fftw_complex *u0x, int nu0x, fftw_complex *u0y, int nu0y, fftw_complex *u1x, int nu1x, fftw_complex *u1y, int nu1y, double usefulStuff[10], double dt, double dz, int nz, int nalphaa, double *alphaa, int nalphab, double *alphab, int nbetaa, double *betaa, int nbetab, double *betab, double gamma, double psp, int maxiter, double tol, double psi, double chi, int verbose);


