/*****************************************************************
*	
*			Header for the plain C version of ssprop
*
******************************************************************/

#include "fftw3.h"

#define REAL double
#define COMPLEX fftw_complex

void scalar(fftw_complex *u_ini, int nt_ini, fftw_complex *u1, int nt_1, double usefulStuff[10], double dt, double dz, int nz, int nalpha, double *alphap, int nbeta, double *beta, double gamma, double traman, double toptical, int maxiter, double tol, int verbose);
