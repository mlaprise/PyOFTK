/*****************************************************************
*	
*			Header for the plain C version of ssprop
*
******************************************************************/

#include "fftw3.h"

#define REAL double
#define COMPLEX fftw_complex

void testNumpy(COMPLEX* a, int na, COMPLEX *b, int nb, COMPLEX* AplusB, int nAplusB);

void vector(COMPLEX *u0x, int nu0x, COMPLEX *u0y, int nu0y, COMPLEX *u1x, int nu1x, COMPLEX *u1y, int nu1y, double usefulStuff[10], REAL dt, REAL dz, int nz, int nalphaa, double *alphaa, int nalphab, double *alphab, int nbetaa, double *betaa, int nbetab, double *betab, REAL gamma, REAL psp, int maxiter, REAL tol, REAL psi, REAL chi,int verbose);


