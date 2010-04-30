#include <math.h>
#include <time.h>
#include <fftw.h>


fftw_complex exp_c( fftw_complex &z );

void swap( fftw_complex &A, fftw_complex &B );

void Prop( fftw_complex *champ_in, int size_champ_x, int size_champ_y, double lambda, double dx, double Ax, double Bx, double Cx, double Dx, double Ay, double By, double Cy, double Dy);


