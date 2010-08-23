/***********************************************
*	Test fkabcd.cpp 
*   
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
#include <iostream>
#include "fkabcd.h"

using std::cout;
using std::endl;

void prtSqMat(fftw_complex *mat, int n)
{
	int i,j;	

	for( i = 0; i < n; i++)
	{
		for( j = 0; j < n; j++)
		{
			cout << mat[(j*n)+i].re << "+" << mat[(j*n)+i].im << "i  ";
		}
		cout << endl;
	}
}


int main()
{
	int i,j;
	int size = 2048;
	double chrono;
	time_t time1;
	time_t time2;
	fftw_complex *test;
	test = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size * size);

	for( i = 0; i < size; i++)
	{
		for( j = 0; j < size; j++)
		{
			test[(i*size)+j].re = i;
			test[(i*size)+j].im = 0.0;
		}
	}
	
	//cout << "Input" << endl;
	//prtSqMat(test, size);

	// ***** CHRONO POUR DEV. ****
	time1=time(NULL);
	// ********** FIN ************

	Prop(test, size, size, 632e-9, 0.0000116, 1.0, 0.50, 0.0, 1.0, 1.0, 0.50, 0.0, 1.0);

	// ***** CHRONO POUR DEV. ****
	time2=time(NULL);
	chrono = difftime(time2, time1);
	cout << "Duree du Calcul: " << chrono << " secondes" << endl;
	// ********** FIN ************

	//cout << "Output" << endl;
	// prtSqMat(test, size);
	
	delete [] test;
	return 0;
}

