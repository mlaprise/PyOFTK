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
import numpy.linalg.linalg as la
from scipy import *
import pulse
from utilities import *
import scipy.fftpack as fftpack



def spectrogram(t, eFieldSVEA, lambdaZero = 0.0):
	'''
	Compute the spectrogram of a SVEA pulse
	'''

	nt = len(t)
	tn = linspace(-nt/2,nt/2, nt)
	gateFunc = pulse.gaussianPulse(tn, 20.0, 0.0, 1.0, 12)
	[w, U] = pulseSpectrum(t, eFieldSVEA, lambdaZero, units='nm')
	nbPoints = len(eFieldSVEA)
	ampFrog = matrix(eFieldSVEA).T*matrix(gateFunc)

	for i in arange(1,nbPoints):
		ampFrog[:,i] = roll(ampFrog[:,i], (nbPoints/2)-i, axis=0)

	ampFrog = fftpack.fftshift(fftpack.ifft(fftpack.ifftshift(ampFrog)))
	intFrog = pow(abs(ampFrog),2)

	return [w,ampFrog]


def genFrog(eFieldSVEA, gateFunc):
	'''
	Compute the FROG trace of a SVEA pulse
	'''
	
	nbPoints = len(eFieldSVEA)
	ampFrog = array(matrix(eFieldSVEA).T*matrix(gateFunc))

	for i in arange(1,nbPoints):
		ampFrog[i,:] = roll(ampFrog[i,:], -i)

	# EF = fftshift(ifft(ifftshift(EF), [], 1), 1);
	ampFrog = fftpack.fftshift( fftpack.ifft( fftpack.ifftshift(ampFrog), axis=0 ), (0,))
	# EF = fftshift(fft(ifftshift(Field, 1), [], 1));
	intFrog = pow(abs(ampFrog),2)

	return [intFrog, ampFrog]


def svdFrog(ampFrog):
	'''
	Extracts the pulse and gate as functions of time from the FROG
	with the singular value decomposition method
	'''

	E =	fftpack.fftshift( fftpack.fft( fftpack.ifftshift(ampFrog, (0,)), axis=0 ) )
	nbPoints = shape(E)[0]

	for i in arange(1,nbPoints):
		E[i,:] = roll(E[i,:], i)

	[U, S, V] = la.svd(E)

	pulseField = U[:,0]
	gateFunction = conj(V[0,:])

	return [pulseField, gateFunction]


def pwmFrog(ampFrog):
	'''
	Extracts the pulse and gate as functions of time from the FROG
	with the power method
	'''

	E =	fftpack.fftshift( fftpack.fft( fftpack.ifftshift(ampFrog, (0,)), axis=0 ) )
	nbPoints = shape(E)[0]

	for i in arange(1,nbPoints):
		E[i,:] = roll(E[i,:], i)

	OTO = array(matrix(E).T*matrix(E))
	OOT = array(matrix(E)*matrix(E).T)

	pulseField = powerMethod(OOT)[0]
	gateFunction = powerMethod(OTO)[0]

	return [pulseField, gateFunction]


def pwmFrogGPU(ampFrog):
	'''
	Extracts the pulse and gate as functions of time from the FROG
	with the power method
	'''

	E =	fftpack.fftshift( fftpack.fft( fftpack.ifftshift(ampFrog, (0,)), axis=0 ) )
	nbPoints = shape(E)[0]

	for i in arange(1,nbPoints):
		E[i,:] = roll(E[i,:], i)

	OTO = array(matrix(E).T*matrix(E))
	OOT = array(matrix(E)*matrix(E).T)

	pulseField = powerMethod(OOT)[0]
	gateFunction = powerMethod(OTO)[0]

	return [pulseField, gateFunction]
	

def normalise(frogTrace):
	return frogTrace / abs(frogTrace).sum().sum()


def chi2(intFrogR, intFrog):
	'''
	Evaluate the error between two images
	with the chi2
	'''

	nbrPoints = shape(intFrog)[0]
	return sqrt( pow((intFrogR-intFrog),2).sum().sum() ) / nbrPoints


def powerMethod(intFrog, itr = 1):
	'''
	Extracts the pulse and gate as functions of time from the FROG
	with the power method
	'''

	nbrPoints = shape(intFrog)[0]
	tn = linspace(-nbrPoints/2, nbrPoints/2, nbrPoints)
	U0 = (pulse.gaussianPulse(tn, nbrPoints/2, 1.0) * (1 + 0.4*rand(nbrPoints))).real

	for i in arange(itr):
		U0A = array(matrix(intFrog)*matrix(U0).T)
		U0 = U0A.T/(la.norm(U0A))

	return U0


def powerMethodGPU(intFrog, itr = 1):
	'''
	Extracts the pulse and gate as functions of time from the FROG
	with the power method
	'''

	nbrPoints = shape(intFrog)[0]
	tn = linspace(-nbrPoints/2, nbrPoints/2, nbrPoints)
	U0 = (pulse.gaussianPulse(tn, nbrPoints/2, 1.0) * (1 + 0.4*rand(nbrPoints))).real

	for i in arange(itr):
		U0A = array(matrix(intFrog)*matrix(U0).T)
		U0 = U0A.T/(la.norm(U0A))

	return U0


def extractFrog(intFrog, epsTol = 1E-5, iterMax = 1000, method='svd', showGraph = 1):
	'''
	Reconstruct the phase with the svd (default) or power method
	'''

	nbrPoints = shape(intFrog)[0]
	tn = linspace(-nbrPoints/2, nbrPoints/2, nbrPoints)
	epsArchive = zeros(iterMax, float)

	# Generate the seed
	pulseSeed = pulse.gaussianPulse(tn, nbrPoints/2, 1.0) * (1 + 0.4*rand(nbrPoints))
	gateSeed =  pulse.gaussianPulse(tn, nbrPoints/2, 1.0) * (1 + 0.4*rand(nbrPoints))

	# Normalise the input FROG trace
	intFrog = normalise(intFrog)

	# Generate a new FROG trace from the seed
	[intFrogR, ampFrogR] = genFrog(pulseSeed, gateSeed)
	intFrogR = normalise(intFrogR)

	# Find chi^2 error
	eps = chi2(intFrog, intFrogR)
	epsArchive[0] = eps
	itr = 0
	
	# Plot init (very slow !)
	if showGraph:
		plt.ion()
		fig = plt.figure(figsize=(12,7))
		ax = fig.add_subplot(231)	
		line1, line2, = pl.plot(tn, pow(abs(pulseSeed),2), tn, pow(abs(gateSeed),2))
		ax2 = fig.add_subplot(232)
		image1 = pl.imshow(intFrogR, interpolation="nearest", aspect="normal")
		ax2.set_title("Reconstructed frog trace")
		ax2c = fig.add_subplot(233)
		image2 = pl.imshow(intFrog, interpolation="nearest", aspect="normal")
		image2.set_array(intFrog)
		ax2c.set_title("Measure frog trace")

	while (eps>epsTol) & (itr<iterMax):

		itr = itr+1
		
		# Find any zero amplitudes
		intFrogR[ where(intFrogR == 0) ] = NaN
		# Normalise amplitudes (keep phase information)
		ampFrogR = ampFrogR*(sqrt(intFrog/intFrogR))
		# Remove divide by zeros
		ampFrogR[where(isnan(intFrogR))] = 0.0
		
		# Compute the next guest
		[pulseSeed, gateSeed]  = {
		  'svd': lambda: svdFrog(ampFrogR),
		  'pwm': lambda: pwmFrog(ampFrogR),
		}[method]()

		# Make a FROG trace from new fields
		[intFrogR, ampFrogR] = genFrog(pulseSeed, gateSeed)
		intFrogR = normalise(intFrogR)
		
		eps = chi2(intFrog, intFrogR)
		epsArchive[itr-1] = eps
		
		# Update plot
		if showGraph:
			p_Int = pow(abs(pulseSeed),2)
			g_Int = pow(abs(gateSeed),2)
			maxValue = array([g_Int.max(), p_Int.max()]).max()
			ax.set_ylim((0,maxValue))
			line1.set_ydata(pow(abs(pulseSeed),2))
			line2.set_ydata(pow(abs(gateSeed),2))
			image1.set_array(intFrogR)
			image1.autoscale()
			ax.set_title("Error: " + str(eps))
			pl.draw()

	gateSeed = normalise(gateSeed)
	pulseSeed = normalise(pulseSeed)

	return [pulseSeed, gateSeed, epsArchive, itr]


def extractFrogGPU(intFrog, epsTol = 1E-5, iterMax = 1000, method='svd', showGraph = 1):
	'''
	Reconstruct the phase with the svd (default) or power method
	'''

	nbrPoints = shape(intFrog)[0]
	tn = linspace(-nbrPoints/2, nbrPoints/2, nbrPoints)
	epsArchive = zeros(iterMax, float)

	# Generate the seed
	pulseSeed = pulse.gaussianPulse(tn, nbrPoints/2, 1.0) * (1 + 0.4*rand(nbrPoints))
	gateSeed =  pulse.gaussianPulse(tn, nbrPoints/2, 1.0) * (1 + 0.4*rand(nbrPoints))

	# Normalise the input FROG trace
	intFrog = normalise(intFrog)

	# Generate a new FROG trace from the seed
	[intFrogR, ampFrogR] = genFrog(pulseSeed, gateSeed)
	intFrogR = normalise(intFrogR)

	# Find chi^2 error
	eps = chi2(intFrog, intFrogR)
	epsArchive[0] = eps
	itr = 0
	
	# Plot init (very slow !)
	if showGraph:
		plt.ion()
		fig = plt.figure(figsize=(12,7))
		ax = fig.add_subplot(231)	
		line1, line2, = pl.plot(tn, pow(abs(pulseSeed),2), tn, pow(abs(gateSeed),2))
		ax2 = fig.add_subplot(232)
		image1 = pl.imshow(intFrogR, interpolation="nearest", aspect="normal")
		ax2.set_title("Reconstructed frog trace")
		ax2c = fig.add_subplot(233)
		image2 = pl.imshow(intFrog, interpolation="nearest", aspect="normal")
		image2.set_array(intFrog)
		ax2c.set_title("Measure frog trace")

	while (eps>epsTol) & (itr<iterMax):

		itr = itr+1
		
		# Find any zero amplitudes
		intFrogR[ where(intFrogR == 0) ] = NaN
		# Normalise amplitudes (keep phase information)
		ampFrogR = ampFrogR*(sqrt(intFrog/intFrogR))
		# Remove divide by zeros
		ampFrogR[where(isnan(intFrogR))] = 0.0
		
		# Compute the next guest
		[pulseSeed, gateSeed]  = {
		  'svd': lambda: svdFrog(ampFrogR),
		  'pwm': lambda: pwmFrogGPU(ampFrogR),
		}[method]()

		# Make a FROG trace from new fields
		[intFrogR, ampFrogR] = genFrog(pulseSeed, gateSeed)
		intFrogR = normalise(intFrogR)
		
		eps = chi2(intFrog, intFrogR)
		epsArchive[itr-1] = eps
		
		# Update plot
		if showGraph:
			p_Int = pow(abs(pulseSeed),2)
			g_Int = pow(abs(gateSeed),2)
			maxValue = array([g_Int.max(), p_Int.max()]).max()
			ax.set_ylim((0,maxValue))
			line1.set_ydata(pow(abs(pulseSeed),2))
			line2.set_ydata(pow(abs(gateSeed),2))
			image1.set_array(intFrogR)
			image1.autoscale()
			ax.set_title("Error: " + str(eps))
			pl.draw()

	gateSeed = normalise(gateSeed)
	pulseSeed = normalise(pulseSeed)

	return [pulseSeed, gateSeed, epsArchive, itr]


def matlab_xcorr(x, y):
	return signal.correlate(x, y, mode='same')
