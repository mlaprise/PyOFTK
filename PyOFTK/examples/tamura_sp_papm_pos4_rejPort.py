#!/usr/bin/env python
__version__ = "0.1"
__date__ = "2009-02-03"
__author__ ="Martin Laprise"


# *********************************************
#	Simulation d'une cavite P-APM Self-Similar
#	mod. de stretched-pulse
#   
#
#
#
#	Author: 	Martin Laprise
#		    	Universite Laval
#				martin.laprise.1@ulaval.ca
#                 
# *********************************************


import numpy
from numpy import *
from scipy import *
from pylab import *
import scipy.interpolate
from scipy import integrate
import PyOFTK
import time


if __name__ == '__main__':

	t1 = time.time()

	lambdaZero = 1.030e-6
	T = 100.0
	nt = pow(2,12)
	dt = T/float(nt)
	t = linspace(-T/2, T/2, nt)
	w = PyOFTK.wspace(T,nt)
	vs = fftpack.fftshift(w/(2*pi))
	C = 2.99792458e-4
	nbSection = 4
	nbPass =  512
	
	dz = zeros(nbSection)
	nz = zeros(nbSection, int)
	gamma = zeros(nbSection)
	betap = zeros([nbSection, 4], float)
	alpha = zeros([nbSection, 1], float)
	alphaSat = array([0.0])
	alphaProfil = zeros(nt, float)
	pulseEnergy = zeros(nbPass, float)
	phiNLArchive = zeros(nbPass, float)
	#pulseWidth = zeros(nbPass, float)
	#spectralWidth = zeros(nbPass, float)
	#pulsePeak = zeros(nbPass, float)
	#pulseCenter = zeros(nbPass, float)

	fibreYb = PyOFTK.FibreStepIndex(2.5, 62.5, 0.0435, 0, 1)
	fibreFlexCore = PyOFTK.FibreStepIndex(2.5, 62.5, 0.0435, 0, 1)

	# Reseau
	dz[0] = 0.4
	nz[0] = 10

	# Flexcore
	dz[1] = 0.045
	nz[1] = 50

	# Fibre Dopee
	dz[2] = 0.012
	nz[2] = 25

	# Tit boute de Flexcore
	dz[3] = 0.0002
	nz[3] = 1

	# Couplage externe en amplitude (sqrt(couplage en intensite))
	extCoupling = sqrt(0.01)
	
	# Bande passante en nm
	gainBandwidth = 40
	energySat = 1500
	# Parametres de l'absorbant saturable lent
	recoveryTime = 0.5
	nonSatLoss = 0.16
	satInt = 200
	depMod = 0.3

	# Periode
	periodeCavite = (dz[0]*nz[0]/(2.997e8) + dz[1]*nz[1]/(2.997e8/fibreFlexCore.effIndex(1.03)) + dz[3]*nz[3]/(2.997e8/fibreYb.effIndex(1.03))) + dz[2]*nz[2]/(2.997e8/fibreFlexCore.effIndex(1.03))
	print "Periode: " + str(periodeCavite*1e9) + " ns"
	print "Taux de repetition: " + str((1e-6)/periodeCavite) + " Mhz"

	# Dispersion (beta2) [ps2/m]
	print "Longueur du reseau: " +  str(dz[0]*nz[0]) + " m"
	betap[0] = array([0,0,-0.015,0])
	alpha[0] = array([0.0])

	print "Longueur de la fibre 2: " +  str(dz[1]*nz[1]) + " m"
	betap[1] = array([0,0,fibreFlexCore.beta2(1.03)*1e24,0])
	alpha[1] = array([0.0])

	print "Longueur de la fibre dopee: " +  str(dz[2]*nz[2]) + " m"
	betap[2] = array([0,0,fibreYb.beta2(1.03)*1e24,0])
	alpha[2] = array([-5.0])
	print "Parametre de gain alpha: " + str(alpha[2][0])
	print "Parametre de gain g: " + str(exp(-alpha[2][0]*nz[2]*dz[2]))
	print "Parametre de gain g[dB/m]: " + str(10*log10(exp(-alpha[2][0]*1))) + " dB/m"


	print "Longueur de la fibre 4: " +  str(dz[3]*nz[3]) + " m"
	betap[3] = array([0,0,fibreFlexCore.beta2(1.03)*1e24,0])
	alpha[3] = array([0.0])


	print "Dispersion globale: " + str(betap[0][2]*dz[0]*nz[0] + betap[1][2]*dz[1]*nz[1] + betap[2][2]*dz[2]*nz[2] + betap[3][2]*dz[3]*nz[3]) + " ps2"


	# u_ini_x_non_chirpe = numpy.random.normal(loc=0.0, scale=0.01, size=nt)
	u_ini_x_non_chirpe = PyOFTK.gaussianPulse(t,20,0,5,4,0)*numpy.random.normal(loc=0.0, scale=0.01, size=nt)
	u_ini_y_non_chirpe = zeros(nt)
	archive = zeros([nbPass+1, nt], complex)
	archiveInt = zeros([nbPass+1, nt], double)
	archiveIntOutput = zeros([nbPass+1, nt], double)
	archiveSpectreOutput = zeros([nbPass+1, nt], double)
	archivePass= zeros([7, nt], complex)
	absSatArchive = zeros([nbPass+1, nt], double)
	[u_ini_x, u_ini_y, outputParam] = PyOFTK.vector(u_ini_x_non_chirpe, u_ini_y_non_chirpe, dt, dz[0], nz[0], alpha[0], alpha[0], betap[0], betap[0], 0.0065169, 0.0, 50, 1e-5, 0, 0)

	# Seed avec des resultats precedents	
	# u_out_x3 = 0.01*PyOFTK.hdf5load("caviteml31.h5")[1022]
	# u_ini_x = PyOFTK.satAbsorber(u_out_x3, satInt, depMod)
	u_ini_x = u_ini_x_non_chirpe

	archive[0] = u_ini_x
	archiveInt[0] = pow(abs(u_ini_x),2)
	
	# Compute the lambda value of the spectrum
	wavelength = (1/((vs/C)+1/(lambdaZero)))*1e9


	# *********************************************
	#			   Loop principale 
	# *********************************************
	for i in arange(nbPass):

		phiNL = 0

		### Passe dans l'absorbant saturable rapide ###
		u_out_x = PyOFTK.satAbsorber(u_ini_x, satInt, depMod)
		u_out_y = PyOFTK.satAbsorber(u_out_x, satInt, depMod)
		archivePass[0] = u_out_x
		rejPort = (u_ini_x - u_out_x)

		### Reseau de dispersion ###
		[u_out_x2, u_out_y2, outputParam] = PyOFTK.vector(u_out_x, u_out_y, dt, dz[0], nz[0], alpha[0], alpha[0], betap[0], betap[0], 0.0, 0.0, 50, 1e-5, 0, 0)
		phiNL = phiNL + outputParam[0]
		archivePass[1] = u_out_x2

		### FlexCore ###
		[u_out_x3, u_out_y3, outputParam] = PyOFTK.vector(u_out_x2, u_out_y2, dt, dz[1], nz[1], alpha[1], alpha[1], betap[1], betap[1], fibreFlexCore.nlGamma(1.03), 0.0, 50, 1e-5, 0, 0)
		phiNL = phiNL + outputParam[0]
		archivePass[2] = u_out_x3

		### Section de fibre dopee ###
		# Make the alpha parameter satured with the total energy
		alphaSatPeak = alpha[2]/(1.0 + pulseEnergy[i-1]/energySat)
		# Reconstruct the lorenztian gain profil with an homogeneous saturation
		alphaSat = PyOFTK.gainProfil(wavelength, alphaSatPeak, gainBandwidth)
		[u_out_x4, u_out_y4, outputParam] = PyOFTK.vector(u_out_x3, u_out_y3, dt, dz[2], nz[2], fftpack.fftshift(alphaSat), fftpack.fftshift(alphaSat), betap[2], betap[2], fibreYb.nlGamma(1.03), 0.0, 50, 1e-5, 0, 0)
		phiNL = phiNL + outputParam[0]
		archivePass[3] = u_out_x4

		### Flexcore 0.3m	###
		[u_out_x5, u_out_y5, outputParam] = PyOFTK.vector(u_out_x4, u_out_y4, dt, dz[3], nz[3], alpha[3], alpha[3], betap[3], betap[3], fibreFlexCore.nlGamma(1.03), 0.0, 50, 1e-5, 0, 0)
		phiNL = phiNL + outputParam[0]
		archivePass[4] = u_out_x5

		# Coupleur de sortie
		u_ini_x = (1-extCoupling)*u_out_x5
		u_ini_y = (1-extCoupling)*u_out_y5
		archive[i] = extCoupling*u_out_x5
		archiveInt[i] = pow(abs((1-extCoupling)*u_out_x5),2)
		archiveIntOutput[i] = pow(abs(extCoupling*u_out_x5),2)
		archiveSpectreOutput[i] = fftpack.fftshift(pow(abs(dt*fftpack.fft(extCoupling*u_out_x5)/sqrt(2*pi)),2))
		archivePass[5] = u_ini_x

		pulseEnergy[i] = dt*archiveInt[i].sum()
		phiNLArchive[i] = phiNL
		spectre = fftpack.fftshift(pow(abs(dt*fftpack.fft(u_out_x5)/sqrt(2*pi)),2))



	spectre_out = archiveSpectreOutput[511]
	spectre_in = fftpack.fftshift(pow(abs(dt*fftpack.fft(archive[0,:])/sqrt(2*pi)),2))

	bandwidth = 1000*(wavelength[nt-1] - wavelength[0])

	print "Energie port 1: "+str(dt*pow(abs(extCoupling*u_out_x5),2).sum())+" pJ"
	print "Energie port 2 (rejection port): "+str(dt*pow(abs(rejPort),2).sum())+" pJ"


	print "Phase non-lineaire accumule par tours: " + str(phiNL/pi) + "pi"

	# Compute the chirp of the pulse
	pulsePeak = archiveInt[nbPass-1].max()
	pulseCenter = where(archiveInt[nbPass-1] == pulsePeak)[0][0]
	nu_inst_out = PyOFTK.nuInst(u_ini_x_non_chirpe, pulseCenter)
	nu_inst_out2 = PyOFTK.nuInst(u_out_x2, pulseCenter)
	nu_inst_out3 = PyOFTK.nuInst(extCoupling*u_out_x5, pulseCenter)

	# Reseau pour recompresser
	[u_out_x_recompresse, u_out_y_recompresse, outputParam] = PyOFTK.vector(extCoupling*u_out_x5, extCoupling*u_out_y5, dt, 0.065, 50, alpha[1], alpha[1], -betap[1], -betap[1], 0, 0.0, 50, 1e-5, 0, 0)

	nu_inst_out4 = PyOFTK.nuInst(u_out_x_recompresse, pulseCenter)

	# Bench
	elapsed = time.time() - t1
	print "Processing time: " + str(elapsed) + " secondes" + "(" + str(elapsed/60) + " minutes)"

	# Calcul de l'evolution du profil dans une passe
	section = 1
	u_pass_x = zeros([nz[section], nt], complex)
	u_pass_y = zeros([nz[section], nt], complex)
	u_pass_x_int = zeros([nz[section], nt], double)
	[u_pass_x[0], u_pass_y[0], outputParam] = PyOFTK.vector(archivePass[section], u_out_y2, dt, dz[section], 1, alpha[section], alpha[section], betap[section], betap[section], fibreFlexCore.nlGamma(1.03), 0.0, 50, 1e-5, 0, 0)
	u_pass_x_int[0] = pow(abs(u_pass_x[0]),2)
	for i in arange(1,nz[section]):
		[u_pass_x[i], u_pass_y[i], outputParam] = PyOFTK.vector(archivePass[section], u_out_y2, dt, dz[section], i, alpha[section], alpha[section], betap[section], betap[section], fibreFlexCore.nlGamma(1.03), 0.0, 50, 1e-5, 0, 0)
		u_pass_x_int[i] = pow(abs(u_pass_x[i]),2)

	# Graph
	figure(figsize=(12,9))

	ax3 = subplot(221)
	plot(t, pow(abs(extCoupling*u_out_x5),2), color="black")
	ylabel("$|u(z,T)|^2$")
	xlabel("$T/T_0$")
	xlim([-T/2,T/2])
	grid(True)
	ax4 = twinx()
	plot(t[0:nt-1], nu_inst_out3)
	ylabel("Chirp")
	ax4.yaxis.tick_right()
	ylim([-1,1])

	ax5 = subplot(223)
	plot(t, pow(abs(u_out_x_recompresse),2), color="black")
	ylabel("$|u(z,T)|^2$")
	xlabel("$T/T_0$")
	xlim([-T/2,T/2])
	grid(True)
	ax4 = twinx()
	plot(t[0:nt-1], nu_inst_out4)
	ylabel("Chirp")
	ax4.yaxis.tick_right()
	ylim([-1,1])


	ax7 = subplot(222)
 	plot(wavelength, spectre_out, color="black")
	grid(True)

	ax8 = subplot(224)
 	semilogy(wavelength, spectre_out, color="black")
	grid(True)

	show()


