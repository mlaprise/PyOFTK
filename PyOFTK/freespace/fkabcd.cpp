/*

Copyright (C) 2007-2010 Martin Laprise (mlapris@gmail.com)

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; version 2 dated June, 1991.

This software is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANDABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software Foundation,
Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA


*/


/***********************************************
*	Resolve the Fresnel-Kirckoff integral for 
*   an abritrary ABCD paraxial system
*
*
*
*
*	Author: 	Martin Laprise
*		    	Universite Laval
*				martin.laprise.1@ulaval.ca
*                
*************************************************/


#include <math.h>
#include <time.h>
#include <fftw.h>
#include "fkabcd.h"


fftw_complex exp_c( fftw_complex &z )
{
	fftw_complex var_comp;
	var_comp.re = exp(z.re) * cos(z.im);
	var_comp.im = exp(z.re) * sin(z.im);
	return var_comp; 

}


void swap( fftw_complex &A, fftw_complex &B )
{
	
	fftw_complex tamp;

	tamp = B;
	B = A;
	A = tamp;
	
}


void Prop( fftw_complex *champ_in, int size_champ_x, int size_champ_y, double lambda, double dx, double Ax, double Bx, double Cx, double Dx, double Ay, double By, double Cy, double Dy)
{
	// Declaration de variable divers
	double dnu_x;
	double size_champ_carre = static_cast<double>(size_champ_x, size_champ_y);
	double deux = 2;
	double puissance1, puissance2;
	const double pi = 3.14159265358979323846264338327950288;
	double y,nu_x,nu_y;
	int i,j;
	
		
	// Declarations des vecteurs de variables complexes
	fftw_complex temp1, temp2, t_f_i_el;


	//***********************************
	// Allocation dynamique de la memoire
	//***********************************


	// Facteur de phase
	fftw_complex **phase_factor1;
	phase_factor1 = new fftw_complex* [size_champ_x];
	for ( i = 0; i < size_champ_x; i++ )
	{
	  phase_factor1[i]=new fftw_complex [size_champ_y];
	}
	
	
	// Recipient de la premiere TF
	fftw_complex *t_f = new fftw_complex [size_champ_x*size_champ_y];


	
	// Recipient de la deuxime TF (inverse)
	fftw_complex *t_f_i = new fftw_complex [size_champ_x*size_champ_y];
	
	// Vecteur de position x
	double *x = new double [size_champ_x];

	//***********************************
	
		
	// Creation du plan pour la FFTW
	fftwnd_plan p1;
	fftwnd_plan p2;
	
	
	// ************************************************************************************************
	// ************************************  Calculs de la BPM 2D  ************************************
   	// ************************************************************************************************

	p1 = fftw2d_create_plan(size_champ_x, size_champ_y, FFTW_FORWARD, FFTW_ESTIMATE | FFTW_IN_PLACE);
	p2 = fftw2d_create_plan(size_champ_x, size_champ_y, FFTW_BACKWARD, FFTW_ESTIMATE | FFTW_IN_PLACE);
  	dnu_x = 1/( static_cast< double >( size_champ_x )*dx );
	
	
	
	// Creation du vecteur d'entree a partir de dx
	for( i =  0; i < size_champ_x; i++ )
	{
	
		x[i] = static_cast< double >( -(size_champ_x/2) + i)*dx; 
		
	}
	


	/*
	cout << "\n\nAmplitude du champ de initiale:" << endl;
	for( i = 0; i < size_champ_x; i++)
	{
		cout << endl;
		for( j = 0; j < size_champ_y; j++)
		{
			cout << champ_in[i][j].re << " +i" << champ_in[i][j].im << "   ";
		}
	
	}
	*/
	

	// Calculs du premier terme de phases
	temp1.re = 0;
	temp2.re = 0;
	for( i = 0; i < size_champ_x; i++)
	{
		for( j = 0; j < size_champ_y; j++)
		{
				
			y = static_cast< double >( -(size_champ_y/2) + j)*dx;
			puissance1 = pow(x[i], deux);
			puissance2 = pow(y,deux);
			temp1.im = (-pi*(Ax-1)*puissance1)/(lambda*Bx );
			temp2.im = (-pi*(Ay-1)*puissance2)/(lambda*By );
			phase_factor1[i][j].re = ( exp_c(temp1).re * exp_c(temp2).re ) - ( exp_c(temp1).im * exp_c(temp2).im );
			phase_factor1[i][j].im = ( exp_c(temp1).re * exp_c(temp2).im ) + ( exp_c(temp1).im * exp_c(temp2).re );
		}
	}

	
	
	// Multiplication du premier terme de phase par le champ d'entree
	for( i = 0; i < size_champ_x; i++)
	{
		for( j = 0; j < size_champ_y; j++)
		{
			t_f[(i*size_champ_x)+ j].re = (phase_factor1[i][j].re * champ_in[(i*size_champ_x)+ j].re) - (phase_factor1[i][j].im * champ_in[(i*size_champ_x)+ j].im);
			t_f[(i*size_champ_x)+ j].im = (phase_factor1[i][j].re * champ_in[(i*size_champ_x)+ j].im) + (phase_factor1[i][j].im * champ_in[(i*size_champ_x)+ j].re);			
		}
	}
	
	
	// Premiere Transforme de Fourier "in place"
	fftwnd_one(p1, &t_f[0], NULL);
	
	

	// Calculs du second terme de phases
	for( i = 0; i < size_champ_x; i++)
	{
		nu_x = static_cast< double >( -(size_champ_x/2) + i)*dnu_x;
		for( j = 0; j < size_champ_y; j++)
		{
				
			nu_y = static_cast< double >( -(size_champ_y/2) + j)*dnu_x;
			puissance1 = pow(nu_x, deux);
			puissance2 = pow(nu_y, deux);
			temp1.im = pi*lambda*Bx*puissance1;
			temp2.im = pi*lambda*By*puissance2;
			phase_factor1[i][j].re = exp_c(temp1).re * exp_c(temp2).re - exp_c(temp1).im * exp_c(temp2).im;
			phase_factor1[i][j].im = exp_c(temp1).re * exp_c(temp2).im + exp_c(temp1).im * exp_c(temp2).re;
		}
	}
	


	
	// *********************************************************************
	// ifftshift (voir fonction Matlab) du vecteur phase_factor2
	// *********************************************************************
		

	// Swap du quadrant 1 et 3
	for( i = 0; i < (size_champ_x/2); i++)
	{
		for( j = 0; j < (size_champ_y/2); j++)
		{
			swap( phase_factor1[i][j], phase_factor1[i+(size_champ_x/2)][j+(size_champ_y/2)]);
		}
	}
	
	// Swap du quadrant 2 et 4
	for( i = (size_champ_x/2); i < size_champ_x; i++)
	{
		for( j = 0; j < size_champ_y/2; j++)
		{
			swap( phase_factor1[i][j], phase_factor1[i-(size_champ_x/2)][j+(size_champ_y/2)]);
		}
	}
	
	
	// *********************************************************************
	// ***************************  fin  ifftshift *************************
	// *********************************************************************
	

	/*
	cout << "\n\nAmplitude du second de phase 2 swaper:" << endl;
	for( i = 0; i < size_champ_x; i++)
	{
		
		for( j = 0; j < size_champ_y; j++)
		{
			cout << phase_factor2[i][j].re << " +i" << phase_factor2[i][j].im << "   ";
		}
		cout << endl;
	}
	*/

	
	// Multiplication du second terme de phase par la premiere TF
	for( i = 0; i < size_champ_x; i++)
	{
		for( j = 0; j < size_champ_y; j++)
		{
			t_f_i[(i*size_champ_x)+ j].re = phase_factor1[i][j].re * t_f[(i*size_champ_x)+ j].re - phase_factor1[i][j].im * t_f[(i*size_champ_x)+ j].im;
			t_f_i[(i*size_champ_x)+ j].im = phase_factor1[i][j].re * t_f[(i*size_champ_x)+ j].im + phase_factor1[i][j].im * t_f[(i*size_champ_x)+ j].re;			
		}
	}
	
	
	
	// Plus besoin du recipient t_f
	delete t_f;
	
	// Seconde Transforme de Fourier (inverse) "in place"
	fftwnd_one(p2, &t_f_i[0], NULL);

	
	// Calculs du troisieme terme de phases
	for( i = 0; i < size_champ_x; i++)
	{
		for( j = 0; j < size_champ_y; j++)
		{
				
			y = static_cast< double >( -(size_champ_y/2) + j)*dx;
			puissance1 = pow(x[i], deux);
			puissance2 = pow(y, deux);
			temp1.im = (-pi*(Dx-1)*puissance1)/(lambda*Bx);
			temp2.im = (-pi*(Dy-1)*puissance2)/(lambda*By);
			phase_factor1[i][j].re = exp_c(temp1).re * exp_c(temp2).re - exp_c(temp1).im * exp_c(temp2).im;
			phase_factor1[i][j].im = exp_c(temp1).re * exp_c(temp2).im + exp_c(temp1).im * exp_c(temp2).re;
		}
	}
	
	
	
	// Multiplication du troisieme terme de phase par la seconde TF (inverse)
	// (Calculs du champ de sortie finale)
	size_champ_carre = static_cast<double>(size_champ_x * size_champ_y);
	for( i = 0; i < size_champ_x; i++)
	{
		for( j = 0; j < size_champ_y; j++)
		{

			t_f_i_el.re = t_f_i[(i*size_champ_x)+ j].re / size_champ_carre;
			t_f_i_el.im = t_f_i[(i*size_champ_x)+ j].im / size_champ_carre;
			champ_in[(i*size_champ_x)+ j].re = phase_factor1[i][j].re * t_f_i_el.re - phase_factor1[i][j].im * t_f_i_el.im;
			champ_in[(i*size_champ_x)+ j].im = phase_factor1[i][j].re * t_f_i_el.im + phase_factor1[i][j].im * t_f_i_el.re;			
		}
	}
	
	
	//***********************************
	// Desallocation de la memoire
	//***********************************
	
	// Desallocation du Facteur de phase (phase_factor1)
	for ( i = 0; i < size_champ_x	; i++ )
	{
 	 delete [] phase_factor1[i];
	}
	delete [] phase_factor1;
	
	// Plus besoin du recipient t_f_i
	delete t_f_i;	

	//***********************************
	
	

	// Liberation de la memoire pris par p1, p2
	fftwnd_destroy_plan(p1);
	fftwnd_destroy_plan(p2);
	
	
	
	// ************************************************************************************************
	// ********************************* Fin Calculs de la BPM 2D  ************************************
   	// ************************************************************************************************
}
