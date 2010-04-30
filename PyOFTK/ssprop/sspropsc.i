/* *********************************************
#	SWIG Interface to sspropsc_plain.c 
#
#
#
#
#	Author: 	Martin Laprise
#		    	Universite Laval
#				martin.laprise.1@ulaval.ca
#                 
# ********************************************* */


%module sspropsc

%{
	#define SWIG_FILE_WITH_INIT
	#define REAL double
	#define COMPLEX fftw_complex
%}

%{
	#include<math.h>
	#include"sspropsc.h"
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

%apply (fftw_complex* IN_ARRAY1, int DIM1) {(fftw_complex* u_ini, int nt_ini)};
%apply (fftw_complex* ARGOUT_ARRAY1, int DIM1) {(fftw_complex *u1, int nt_1)};
%apply (double ARGOUT_ARRAY1[ANY]) {(double usefulStuff[10])};

%apply (int DIM1, double* IN_ARRAY1) {(int nalpha, double* alphap)};
%apply (int DIM1, double* IN_ARRAY1) {(int nbeta, double* beta)};

extern void scalar(fftw_complex  *u_ini, int nt_ini, fftw_complex  *u1, int nt_1, double usefulStuff[10], double dt, double dz, int nz, int nalpha, double *alphap, int nbeta, double *beta, double gamma, double traman, double toptical, int maxiter, double tol, int verbose);


