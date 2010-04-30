/*  File:           sspropsc_plain.c
 *  Author:         Thomas E. Murphy (tem@umd.edu)
 *  		  		Modified by Martin Laprise ("deMatlabized" it)
 *		   			(martin.laprise.1@ulaval.ca)
 *
 *  Created:        1/17/2001
 *  Modified:       20/07/2006
 *  Version:        2.1.0
 *  Description:    This file solves the nonlinear Schrodinger
 *                  equation for propagation in an optical fiber
 *                  using the split-step Fourier method described
 *                  in "Nonlinear Fiber Optics" (G. Agrawal, 2nd
 *                  ed, Academic Press, 1995, Chapter 2).  The
 *                  routine is compiled as a Matlab MEX program,
 *                  which can be invoked directly from Matlab.
 *                  The code makes extensive use of the fftw
 *                  routines, which can be downloaded from
 *                  http://www.fftw.org/, for computing fast
 *                  Fourier transforms.  The corresponding m-file
 *                  (sspropc.m) provides information on how to
 *                  call this routine from Matlab.
 */


/*****************************************************************

    Copyright 2006, Thomas E. Murphy

    This file is part of SSPROP.

    SSPROP is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version
    2 of the License, or (at your option) any later version.

    SSPROP is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public
    License along with SSPROP; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
    02111-1307 USA

*****************************************************************/

#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "fftw3.h"
#include "sspropvc.h"


using std::cout;
using std::endl;


#define PLAN fftw_plan
#define MAKE_PLAN fftw_plan_dft_1d
#define DESTROY_PLAN fftw_destroy_plan
#define EXECUTE fftw_execute
#define IMPORT_WISDOM fftw_import_wisdom_from_file
#define EXPORT_WISDOM fftw_export_wisdom_to_file
#define FORGET_WISDOM fftw_forget_wisdom
#define WISFILENAME "fftw-wisdom.dat"


#define abs2(x) ((*x)[0] * (*x)[0] + (*x)[1] * (*x)[1])
#define prodr(x,y) ((*x)[0] * (*y)[0] + (*x)[1] * (*y)[1])
#define prodi(x,y) ((*x)[0] * (*y)[1] - (*x)[1] * (*y)[0])
#define round(x) ((int)(x+0.5))
#define pi 3.1415926535897932384626433832795028841972

int i;
// =1 when sspropc first invoked */
static int firstcall = 1;
// =1 when memory is allocated */      		
int allocated = 0;
// planner method */              	
static int method = FFTW_PATIENT;
// static int method = FFTW_EXHAUSTIVE;
// plans for fft and ifft */	
PLAN p1,p2,ip1,ip2;
// these vectors are workspace vectors used in performing the calculations      	
COMPLEX *ufft, *uhalf, *uv, *u1, *halfstep, *u0;

                    	
void sspropc_destroy_data(void);
void sspropc_save_wisdom(void);
void sspropc_initialize_data(int nt, COMPLEX *u0, int verbose = 1);
void cmult(int, COMPLEX*, COMPLEX*, COMPLEX*);
void cscale(int, COMPLEX*, COMPLEX*, REAL);
int ssconverged(int, COMPLEX*, COMPLEX*, REAL);


void sspropc_destroy_data(void)
{
  if (allocated) {
    DESTROY_PLAN(p1);
    DESTROY_PLAN(p2);
    DESTROY_PLAN(ip1);
    DESTROY_PLAN(ip2);
    
    fftw_free(ufft);
    fftw_free(uhalf);
    fftw_free(uv);
    fftw_free(halfstep);
    //fftw_free(u0);
    fftw_free(u1);
    
    allocated = 0;
  }
}

void sspropc_save_wisdom(void)
{
  FILE *wisfile;
  
  wisfile = fopen(WISFILENAME, "w");
  if (wisfile) {
    //cout << "Exporting FFTW wisdom (file = " << WISFILENAME << ")" << endl;
    EXPORT_WISDOM(wisfile);
    fclose(wisfile);
  } 
}

void sspropc_load_wisdom(void)
{
  FILE *wisfile;
  
  wisfile = fopen(WISFILENAME, "r");
  if (wisfile) {
	//printf("Importing FFTW wisdom (file = %s).\n", WISFILENAME);
	IMPORT_WISDOM(wisfile);
	fclose(wisfile);
  }
}

void sspropc_initialize_data(int nt, COMPLEX *u0, int verbose)
{
   
  /* wisdom file */
  FILE* wisfile;
  

  if (firstcall){
	sspropc_load_wisdom();
    firstcall = 0;
  }

  
  ufft = (COMPLEX*) fftw_malloc(sizeof(COMPLEX) * nt);
  uhalf = (COMPLEX*) fftw_malloc(sizeof(COMPLEX) * nt);
  uv = (COMPLEX*) fftw_malloc(sizeof(COMPLEX) * nt);
  halfstep = (COMPLEX*) fftw_malloc(sizeof(COMPLEX) * nt);
  
  if(verbose==1)
  {
  	cout << "Creating FFTW plans (length = " << nt << ") ... " << endl;
  }
  p1 = MAKE_PLAN(nt, u0, ufft, FFTW_FORWARD, method);
  p2 = MAKE_PLAN(nt, uv, uv, FFTW_FORWARD, method);
  ip1 = MAKE_PLAN(nt, uhalf, uhalf, FFTW_BACKWARD, method);
  ip2 = MAKE_PLAN(nt, ufft, uv, FFTW_BACKWARD, method);
  if(verbose==1)
  {
  	cout << "done" << endl;
  }

  allocated = 1;
}

/* computes a = b.*c for complex length-nt vectors a,b,c */
void cmult(int nt, COMPLEX* a, COMPLEX* b, COMPLEX* c)
{
  int jj;

  for (jj = 0; jj < nt; jj++) {
    a[jj][0] = b[jj][0] * c[jj][0] - b[jj][1] * c[jj][1];
    a[jj][1] = b[jj][0] * c[jj][1] + b[jj][1] * c[jj][0];
  }
}

/* assigns a = factor*b for complex length-nt vectors a,b */
void cscale(int nt, COMPLEX* a, COMPLEX* b, REAL factor)
{
  int jj;

  for (jj = 0; jj < nt; jj++) {
    a[jj][0] = factor*b[jj][0];
    a[jj][1] = factor*b[jj][1];
  }
}

int ssconverged(int nt, COMPLEX* a, COMPLEX* b, REAL t)
{
  int jj;
  REAL num, denom;

  for (jj = 0, num = 0, denom = 0; jj < nt; jj++) {
    denom += b[jj][0] * b[jj][0] + b[jj][1] * b[jj][1];
    num += (b[jj][0] - a[jj][0]/nt)*(b[jj][0] - a[jj][0]/nt) + (b[jj][1] - a[jj][1]/nt)*(b[jj][1] - a[jj][1]/nt);
  }
  return (num/denom < t);
}

// void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
void scalar(COMPLEX *u_ini, int nt_ini, COMPLEX *u1, int nt_1, double usefulStuff[10], REAL dt, REAL dz, int nz, int nalpha, double *alphap, int nbeta, double *beta,
	 REAL gamma, REAL traman, REAL toptical, int maxiter, REAL tol, int verbose)
{		
	
  REAL scale;        /* scale factor */

 // REAL dt;           /* time step */
 // REAL dz;           /* propagation stepsize */
 // int nz;            /* number of z steps to take */
 // int nalpha;        /* number of beta coefs */
 // double* alphap;    /* alpha(w) array, if applicable */
 // int nbeta;         /* number of beta coefs */
 // double* beta;      /* dispersion polynomial coefs */
 // REAL gamma;        /* nonlinearity coefficient */
 // REAL traman = 0;   /* Raman response time */
 // REAL toptical = 0; /* Optical cycle time = lambda/c */
 // int maxiter = 4;   /* max number of iterations */
 // REAL tol = 1e-5;   /* convergence tolerance */

  //REAL* w;           /* vector of angular frequencies */
  int nt = nt_ini;
  REAL *w = new REAL [nt];

  int iz,ii,jj;      					/* loop counters */
  REAL phase, alpha, wii, fii;      	 		/* temporary variables */
  COMPLEX
    nlp,           					/* nonlinear phase */
    *ua, *ub, *uc, *u0;  				/* samples of u at three adjacent times */
  char argstr[100];					/* string argument */

  // Initialize the output parametre array
  // usefulStuff[0] --> phiNL
  for(ii=0; ii<10; ii++)
  {
	usefulStuff[ii] = 0;
  }  

/*
  if (nrhs == 1) {
	if (mxGetString(prhs[0],argstr,100)) 
	  mexErrMsgTxt("Unrecognized option.");
	
	if (!strcmp(argstr,"-savewisdom")) {
	  sspropc_save_wisdom();
	}
	else if (!strcmp(argstr,"-forgetwisdom")) {
	  FORGET_WISDOM();
	}
	else if (!strcmp(argstr,"-loadwisdom")) {
	  sspropc_load_wisdom();
	}
	else if (!strcmp(argstr,"-patient")) {
	  method = FFTW_PATIENT;
	}
	else if (!strcmp(argstr,"-exhaustive")) {
	  method = FFTW_EXHAUSTIVE;
	}
	else if (!strcmp(argstr,"-measure")) {
	  method = FFTW_MEASURE;
	}
	else if (!strcmp(argstr,"-estimate")) {
	  method = FFTW_ESTIMATE;
	}
	else
	  mexErrMsgTxt("Unrecognized option.");
	return;
  }

  if (nrhs < 7) 
    mexErrMsgTxt("Not enough input arguments provided.");
  if (nlhs > 1)
    mexErrMsgTxt("Too many output arguments.");
*/

//  sspropc_initialize_data(mxGetNumberOfElements(prhs[0]));
	

	sspropc_load_wisdom();
	
	u0 = (COMPLEX*) fftw_malloc(sizeof(COMPLEX) * nt);
	
	sspropc_initialize_data(nt,u0,verbose);
	sspropc_save_wisdom();
	for(i = 0; i < nt; i++)
	{

		u0[i][0] = u_ini[i][0];
		u0[i][1] = u_ini[i][1];
		
	}

  /* parse input arguments */
//  dt = (REAL) mxGetScalar(prhs[1]);
//  dz = (REAL) mxGetScalar(prhs[2]);
//  nz = round(mxGetScalar(prhs[3]));
//  nalpha = mxGetNumberOfElements(prhs[4]);
//  alphap = mxGetPr(prhs[4]);
//  beta = mxGetPr(prhs[5]);
//  nbeta = mxGetNumberOfElements(prhs[5]);
//  gamma = (REAL) mxGetScalar(prhs[6]);
//  if (nrhs > 7)
//	traman = (mxIsEmpty(prhs[7])) ? 0 : (REAL) mxGetScalar(prhs[7]);
//  if (nrhs > 8)
//	toptical = (mxIsEmpty(prhs[8])) ? 0 : (REAL) mxGetScalar(prhs[8]);
//  if (nrhs > 9)
//	maxiter = (mxIsEmpty(prhs[9])) ? 4 : round(mxGetScalar(prhs[9]));
//  if (nrhs > 10)
//	tol = (mxIsEmpty(prhs[10])) ? 1e-5 : (REAL) mxGetScalar(prhs[10]);
  
 // if ((nalpha != 1) && (nalpha != nt))
 //   mexErrMsgTxt("Invalid vector length (alpha).");

  /* compute vector of angular frequency components */
  /* MATLAB equivalent:  w = wspace(tv); */
  //w = (REAL*)mxMalloc(sizeof(REAL)*nt);
  for (ii = 0; ii <= (nt-1)/2; ii++) {
    w[ii] = 2*pi*ii/(dt*nt);
  }
  for (; ii < nt; ii++) {
    w[ii] = 2*pi*ii/(dt*nt) - 2*pi/dt;
  }

  /* compute halfstep and initialize u1 */

  for (jj = 0; jj < nt; jj++) {
	if (nbeta != nt) 	 
	  for (ii = 0, phase = 0, fii = 1, wii = 1; 
		   ii < nbeta; 
		   ii++, fii*=ii, wii*=w[jj]) 
		phase += wii*((REAL)beta[ii])/fii;
	else
	  phase = (REAL)beta[jj];
	alpha = (nalpha == nt) ? (REAL)alphap[jj] : (REAL)alphap[0];
	halfstep[jj][0] = +exp(-alpha*dz/4)*cos(phase*dz/2);
	halfstep[jj][1] = -exp(-alpha*dz/4)*sin(phase*dz/2);
	//u0[jj][0] = (REAL) mxGetPr(prhs[0])[jj];
	//u0[jj][1] = mxIsComplex(prhs[0]) ? (REAL)(mxGetPi(prhs[0])[jj]) : 0.0;
	u1[jj][0] = u0[jj][0];
	u1[jj][1] = u0[jj][1];
  }

    /* free w vector */
	delete [] w;
	

  if(verbose==1)
  {
	cout << "Performing split-step iterations ... " << endl;
	cout << "Variable traman[0]: " << traman << endl;
  }

  /* ufft = fft(u0) */
  EXECUTE(p1);                          	 
  	
  for (iz = 0; iz < nz; iz++) {
    
    /* uhalf = halfstep.*ufft */
    cmult(nt,uhalf,halfstep,ufft);
	
	/* uhalf = nt*ifft(uhalf) */
    EXECUTE(ip1);

    for (ii = 0; ii < maxiter; ii++) {                
	
      if ((traman == 0.0) && (toptical == 0)) {

        for (jj = 0; jj < nt; jj++) {	
          phase = gamma*(u0[jj][0]*u0[jj][0] + u0[jj][1]*u0[jj][1] + u1[jj][0]*u1[jj][0] + u1[jj][1]*u1[jj][1])*dz/2;
          uv[jj][0] = (uhalf[jj][0]*cos(phase) + uhalf[jj][1]*sin(phase))/nt;
          uv[jj][1] = (-uhalf[jj][0]*sin(phase) + uhalf[jj][1]*cos(phase))/nt;
        }

      } else {

        jj = 0;
        ua = &u0[nt-1]; ub = &u0[jj]; uc = &u0[jj+1];
        nlp[1] = -toptical*(abs2(uc) - abs2(ua) + prodr(ub,uc) - prodr(ub,ua))/(4*pi*dt);
        nlp[0] = abs2(ub) - traman*(abs2(uc) - abs2(ua))/(2*dt) + toptical*(prodi(ub,uc) - prodi(ub,ua))/(4*pi*dt);
        
        ua = &u1[nt-1]; ub = &u1[jj]; uc = &u1[jj+1];
        nlp[1] += -toptical*(abs2(uc) - abs2(ua) + prodr(ub,uc) - prodr(ub,ua))/(4*pi*dt);
        nlp[0] += abs2(ub) - traman*(abs2(uc) - abs2(ua))/(2*dt) + toptical*(prodi(ub,uc) - prodi(ub,ua))/(4*pi*dt);

        nlp[0] *= gamma*dz/2;
        nlp[1] *= gamma*dz/2;

        uv[jj][0] = (uhalf[jj][0]*cos(nlp[0])*exp(+nlp[1]) + uhalf[jj][1]*sin(nlp[0])*exp(+nlp[1]))/nt;
        uv[jj][1] = (-uhalf[jj][0]*sin(nlp[0])*exp(+nlp[1]) + uhalf[jj][1]*cos(nlp[0])*exp(+nlp[1]))/nt;
      
        for (jj = 1; jj < nt-1; jj++) {
          ua = &u0[jj-1]; ub = &u0[jj]; uc = &u0[jj+1];
          nlp[1] = -toptical*(abs2(uc) - abs2(ua) + prodr(ub,uc) - prodr(ub,ua))/(4*pi*dt);
          nlp[0] = abs2(ub) - traman*(abs2(uc) - abs2(ua))/(2*dt) + toptical*(prodi(ub,uc) - prodi(ub,ua))/(4*pi*dt);

          ua = &u1[jj-1]; ub = &u1[jj]; uc = &u1[jj+1];
          nlp[1] += -toptical*(abs2(uc) - abs2(ua) + prodr(ub,uc) - prodr(ub,ua))/(4*pi*dt);
          nlp[0] += abs2(ub) - traman*(abs2(uc) - abs2(ua))/(2*dt) + toptical*(prodi(ub,uc) - prodi(ub,ua))/(4*pi*dt);

          nlp[0] *= gamma*dz/2;
          nlp[1] *= gamma*dz/2;

          uv[jj][0] = (uhalf[jj][0]*cos(nlp[0])*exp(+nlp[1]) + uhalf[jj][1]*sin(nlp[0])*exp(+nlp[1]))/nt;
          uv[jj][1] = (-uhalf[jj][0]*sin(nlp[0])*exp(+nlp[1]) + uhalf[jj][1]*cos(nlp[0])*exp(+nlp[1]))/nt;
        }

        /* we now handle the endpoint where jj = nt-1 */
        ua = &u0[jj-1]; ub = &u0[jj]; uc = &u0[0];
        nlp[1] = -toptical*(abs2(uc) - abs2(ua) + prodr(ub,uc) - prodr(ub,ua))/(4*pi*dt);
        nlp[0] = abs2(ub) - traman*(abs2(uc) - abs2(ua))/(2*dt) + toptical*(prodi(ub,uc) - prodi(ub,ua))/(4*pi*dt);

        ua = &u1[jj-1]; ub = &u1[jj]; uc = &u1[0];
        nlp[1] += -toptical*(abs2(uc) - abs2(ua) + prodr(ub,uc) - prodr(ub,ua))/(4*pi*dt);
        nlp[0] += abs2(ub) - traman*(abs2(uc) - abs2(ua))/(2*dt) + toptical*(prodi(ub,uc) - prodi(ub,ua))/(4*pi*dt);

        nlp[0] *= gamma*dz/2;
        nlp[1] *= gamma*dz/2;

        uv[jj][0] = (uhalf[jj][0]*cos(nlp[0])*exp(+nlp[1]) + uhalf[jj][1]*sin(nlp[0])*exp(+nlp[1]))/nt;
        uv[jj][1] = (-uhalf[jj][0]*sin(nlp[0])*exp(+nlp[1]) + uhalf[jj][1]*cos(nlp[0])*exp(+nlp[1]))/nt;
      }
  		
      EXECUTE(p2);                    	   /* uv = fft(uv) */
      cmult(nt,ufft,uv,halfstep);          /* ufft = uv.*halfstep */
      EXECUTE(ip2);                    	   /* uv = nt*ifft(ufft) */
      if (ssconverged(nt,uv,u1,tol)) {     /* test for convergence */
        cscale(nt,u1,uv,1.0/nt);           /* u1 = uv/nt; */
        break;                             /* exit from ii loop */
      } else {
        cscale(nt,u1,uv,1.0/nt);           /* u1 = uv/nt; */
      }
    }

    if (ii == maxiter)
    {
      usefulStuff[9] = 1;
      if( (verbose==1) || (verbose == 2) )
  	  {
      	cout << "Failed to converge." << endl;
      }
    }

    /* u0 = u1 */
    cscale(nt,u0,u1,1);                    
  }

  if(verbose==1)
  {
 	cout << "done." << endl;
  }

  fftw_free(u0);
  sspropc_destroy_data();
  
}
