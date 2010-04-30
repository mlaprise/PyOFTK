"""

Copyright (C) 2007-2010 Martin Laprise (mlaprise@gmail.com)

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


"""

from numpy import *
from scipy import *
from utilities import *

def pulseResampling(u_in, newSize):
	
	actuSize = len(u_in)
	u_in_fft = fftpack.fft(u_in)
	temp = fftpack.fftshift(u_in_fft)
	u_out_fft = temp[((actuSize/2) - (newSize/2)):((actuSize/2) + (newSize/2))]
	return fftpack.ifft(fftpack.ifftshift(u_out_fft))


def parabolicPulseFit(t, FWHM, T0, P0):
	'''
	Make a parabolic fit on a pulse-like curve	

		* t: 		vector of times at which to compute u
		* P0:		peak intensity of the pulse @ t=t0 (default = 1)
		* t0:		center of pulse (default = 0)
		* FWHM:		full-width at half-intensity of pulse (default = 1)
	'''

	size = len(t)
	scalingFactor = 0.5
	deltaT = t[size/2 + 1]-t[size/2]
	FWHM_Right = where( (t>(FWHM/2)) & (t<(FWHM/2)+deltaT) )
	FWHM_Left = where( (t>(-FWHM/2)-deltaT) & (t<(-FWHM/2)) )
	
	parabolicFit = parabolicPulse(t, P0)

	while abs( parabolicFit[FWHM_Right] - (P0/2) ) > 1:
		if (parabolicFit[FWHM_Right] < (P0/2) ):
			parabolicFit = parabolicPulse(t/scalingFactor, P0)
		else:
			parabolicFit = parabolicPulse(t*scalingFactor, P0)
		scalingFactor = scalingFactor + 0.01

	return parabolicFit


def parabolicPulse(t, P0):
	'''
	Geneate a parabolic envelope pulse

		* t:		vector of times at which to compute u
		* P0:		peak intensity of the pulse @ t=t0 (default = 1)
	'''

	size = len(t)
	output = zeros(size)
	parabolicSection = -pow(t,2)+P0

	for i in arange(size):
		if parabolicSection[i] > 0:
			output[i] = parabolicSection[i]

	return output


def gaussianPulse(t, FWHM, t0, P0 = 1.0, m = 1, C = 0):
	"""
	Geneate a gaussian/supergaussiance envelope pulse

		* field_amp: 	output gaussian pulse envellope (amplitude).
		* t:     		vector of times at which to compute u
		* t0:    		center of pulse (default = 0)
		* FWHM:   		full-width at half-intensity of pulse (default = 1)
		* P0:    		peak intensity of the pulse @ t=t0 (default = 1)
		* m:     		Gaussian order (default = 1)
		* C:     		chirp parameter (default = 0)
	"""

	t_zero = FWHM/sqrt(4.0*log(2.0))
	amp = sqrt(P0)
	real_exp_arg = -pow(((t-t0)/t_zero),2.0*m)/2.0
	euler1 = cos(-C*real_exp_arg)
	euler2 = sin(-C*real_exp_arg)
	return amp*exp(real_exp_arg)*euler1 + amp*exp(real_exp_arg)*euler2*1.0j


def sechPulse(t, FWHM, t0, P0, C):
	'''
	This function computes a hyperbolic secant pulse with the
	specified parameters:
	'''

	T_zero = FWHM/(2*arccosh(sqrt(2)));
	return sqrt(P0)*1/cosh((t-t0)/T_zero)*exp(-1j*C*pow((t-t0),2)/(2*pow(T_zero,2)));


# u = solitonpulse (t,t0,epsilon,N)
def solitonPulse(t,t0,epsilon,N):
	return N*epsilon*(1/cosh(epsilon*(t-t0)))


def sechPulseCmpt(t, p):
	'''
	This function computes a hyperbolic secant pulse with the
	specified parameters:

	- Compacted version - 
	'''

	FWHM = p[0]
	t0 = p[1]
	P0 = p[2]
	C = p[3]
	T_zero = FWHM/(2*arccosh(sqrt(2)));
	return pow( abs(sqrt(P0)*1/cosh((t-t0)/T_zero)*exp(-1j*C*pow((t-t0),2)/(2*pow(T_zero,2)))), 2)


def gaussianPulseCmpt(t, p):
	"""
	Geneate a gaussian/supergaussiance envelope pulse

		* field_amp: 	output gaussian pulse envellope (amplitude).
		* t:     		vector of times at which to compute u
		* t0:    			center of pulse (default = 0)
		* FWHM:   		full-width at half-intensity of pulse (default = 1)
		* P0:    		peak intensity of the pulse @ t=t0 (default = 1)
		* m:     		Gaussian order (default = 1)
		* C:     		chirp parameter (default = 0)

	- Compacted version -
	"""

	FWHM = p[0]
	t0 = p[1]
	P0 = p[2]
	m = 1
	C = p[3]
	t_zero = FWHM/sqrt(4.0*log(2.0))
	amp = sqrt(P0)
	real_exp_arg = -pow(((t-t0)/t_zero),2.0*m)/2.0
	euler1 = cos(-C*real_exp_arg)
	euler2 = sin(-C*real_exp_arg)

	return pow(abs(amp*exp(real_exp_arg)*euler1 + amp*exp(real_exp_arg)*euler2*1.0j),2) 


def parabolicPulseCmpt(t, t0, P0):
	'''
	Geneate a parabolic envelope pulse

		* t:		vector of times at which to compute u
		* P0:		peak intensity of the pulse @ t=t0 (default = 1)
		* t0:		center of pulse (default = 0)
	'''

	size = len(t)
	output = zeros(size)
	parabolicSection = -pow(t-t0,2)+P0

	for i in arange(size):
		if parabolicSection[i] > 0:
			output[i] = parabolicSection[i]

	return output


def parabolicPulseFitCmpt(t, p):
	'''
	Make a parabolic fit on a pulse-like curve	

		* t: 		vector of times at which to compute u
		* P0:		peak intensity of the pulse @ t=t0 (default = 1)
		* t0:		center of pulse (default = 0)
		* FWHM:		full-width at half-intensity of pulse (default = 1)
	'''
	S = p[0]
	t0 = p[1]
	P0 = p[2]
	size = len(t)
	
	return parabolicPulseCmpt(t*p[0], p[1], p[2])


def parabolicPulseFitCmpt2(t, p):
	'''
	Make a parabolic fit on a pulse-like curve	

		* t: 		vector of times at which to compute u
		* P0:		peak intensity of the pulse @ t=t0 (default = 1)
		* t0:		center of pulse (default = 0)
		* FWHM:		full-width at half-intensity of pulse (default = 1)
	'''
	FWHM = p[0]
	t0 = p[1]
	P0 = p[2]
	size = len(t)
	scalingFactor = 0.5
	deltaT = t[size/2 + 1]-t[size/2]
	FWHM_Right = where( (t>(FWHM/2)) & (t<(FWHM/2)+deltaT) )
	FWHM_Left = where( (t>(-FWHM/2)-deltaT) & (t<(-FWHM/2)) )

	parabolicFit = parabolicPulseCmpt(t, t0, P0)

	while abs( parabolicFit[FWHM_Right] - (P0/2) ) > 1:
		if (parabolicFit[FWHM_Right] < (P0/2) ):
			parabolicFit = parabolicPulseCmpt(t/scalingFactor, t0, P0)
		else:
			parabolicFit = parabolicPulseCmpt(t*scalingFactor, t0, P0)
		scalingFactor = scalingFactor + 0.01

	return parabolicFit
