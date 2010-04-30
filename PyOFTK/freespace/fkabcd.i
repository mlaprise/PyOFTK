/***********************************************
#	SWIG Interface to fkabcd.cpp
#
#
#
#
#	Author: 	Martin Laprise
#		    	Universite Laval
#				martin.laprise.1@ulaval.ca
#                 
# **********************************************/


%module fkabcd

%{
	#define SWIG_FILE_WITH_INIT
	#define REAL double
	#define COMPLEX fftw_complex
%}

%{
	#include<math.h>
	#include"fkabcd.h"
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


%apply (fftw_complex* INPLACE_ARRAY2, int DIM1, int DIM2) {(fftw_complex* champ_in, int size_champ_x, int size_champ_y)};

extern void Prop( fftw_complex *champ_in, int size_champ_x, int size_champ_y, double lambda, double dx, double Ax, double Bx, double Cx, double Dx, double Ay, double By, double Cy, double Dy);


