# *********************************************
#	Reseau en transmission avec dispersion
#	positive
#
#
#
#
#	Author: 	Martin Laprise
#		    	Universite Laval
#				martin.laprise.1@ulaval.ca
#                 
# *********************************************

import pylab as pl
import PyOFTK
LMBD=(1.0526+0.00047)/(2*1.45)
LMBD2=0.34485
LNGTH = 100

fbg1 = PyOFTK.apodizedFBG(3.0, 62.5, 0.04, 0.0, 1e-1, 1e-2, LMBD2)
fbg2 = PyOFTK.apodizedFBG(3.0, 62.5, 0.04, 0.0, 1e-1, 5e-3, LMBD2)
fbg3 = PyOFTK.apodizedFBG(3.0, 62.5, 0.04, 0.0, 1e-1, 1e-3, LMBD2)

print "Longueur d'onde de Bragg: " + str(fbg1.braggWavelength) + " um"
wvl1 = pl.linspace(fbg1.braggWavelength-0.00047, fbg1.braggWavelength-0.00050, LNGTH)
wvl2 = pl.linspace(fbg1.braggWavelength+0.010, fbg1.braggWavelength+0.050, LNGTH)

beta2Grating1 = pl.zeros(LNGTH, float)
beta2Grating2 = pl.zeros(LNGTH, float)
beta2Grating3 = pl.zeros(LNGTH, float)
beta3Grating1 = pl.zeros(LNGTH, float)
beta3Grating2 = pl.zeros(LNGTH, float)
beta3Grating3 = pl.zeros(LNGTH, float)

for i in range(LNGTH):
	 beta2Grating1[i] = fbg1.gBeta2(wvl2[i])*1e24
	 beta2Grating2[i] = fbg2.gBeta2(wvl2[i])*1e24
	 beta2Grating3[i] = fbg3.gBeta2(wvl2[i])*1e24

	 beta3Grating1[i] = fbg1.gBeta3(wvl2[i])*1e36
	 beta3Grating2[i] = fbg2.gBeta3(wvl2[i])*1e36
	 beta3Grating3[i] = fbg3.gBeta3(wvl2[i])*1e36

pl.figure(figsize=(8,12))
pl.subplot(2,1,1)
pl.plot(wvl2*1000, beta2Grating1, linestyle='-', color='black')
pl.plot(wvl2*1000, beta2Grating2, linestyle='--', color='black')
pl.plot(wvl2*1000, beta2Grating3, linestyle='-.', color='black')

pl.title(r'FBG en transmission: $\lambda_B = 1000 nm$')
pl.xlabel("Longueur d'onde [nm]")
pl.ylabel(r"$\beta_2^g $ $[ps^2/m]$")
#pl.ylim([-0.05, 0.18])
pl.xlim([1007, 1053])
pl.legend(("$\delta n$ = 1E-2","$\delta n$ = 5E-3","$\delta n$ = 1E-3"))
pl.grid(True)

pl.subplot(2,1,2)
pl.plot(wvl2*1000, beta3Grating1, linestyle='-', color='black')
pl.plot(wvl2*1000, beta3Grating2, linestyle='--', color='black')
pl.plot(wvl2*1000, beta3Grating3, linestyle='-.', color='black')
pl.title(r'FBG en transmission: $\lambda_B = 1000 nm$')
pl.xlabel("Longueur d'onde [nm]")
pl.ylabel(r"$\beta_3^g $ $[ps^3/m]$")
#pl.ylim([-0.05, 0.18])
pl.xlim([1007, 1053])
pl.legend(("$\delta n$ = 1E-2","$\delta n$ = 5E-3","$\delta n$ = 1E-3"))
pl.grid(True)
pl.show()


