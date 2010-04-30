# *********************************************
#	PyOFTK.ossm() test
#	Example from A. de Toroker et al.
#   when gamma = 0
#
#
#
#	Author: 	Martin Laprise
#		    	Universite Laval
#				martin.laprise.1@ulaval.ca
#                 
# *********************************************

from numpy import *
from pylab import *
from PyOFTK.utilities import *
import PyOFTK
import time
                    
t1 = time.time()
storeResults = 0

nbrPoints_z = 10000
# Duree de pulse desire [ps]
dureePulse = 47.6948
puissanceCrete = 478.8
LMBD = 1.03
nt = 24000
positionDetecteur = 4999

### Creation d'un fbg ###
fbgStart = 2000
fbgStop = 8000
# Pas du reseaux
#Longueur_onde_bragg = 1.03238
Longueur_onde_bragg = 1.03105615
#Longueur_onde_bragg = 1.0289438500000001
GrandLambda = Longueur_onde_bragg/(2*1.45)
# Difference d'indice
deltaIndex = 3.695e-3
fbg1 = PyOFTK.apodizedFBG(2.5, 62.5, 0.05, 0.0, 40e-2, deltaIndex, GrandLambda)
#gamma = fbg1.nlGamma(1.03)
gamma = 0.0
detZero = fbg1.detuning(LMBD)
detVec = zeros(nbrPoints_z,double)
detVec[fbgStart:fbgStop] += detZero
kapZero = fbg1.kappa(LMBD)
kapVec = zeros(nbrPoints_z,double)

#iz = arange(1.0,nbrPoints_z+1)/nbrPoints_z
#kapVec[3000:nbrPoints_z-1] = kapZero*(iz[3000:nbrPoints_z-1])
kapVec[fbgStart:fbgStop] = kapZero

# Initialisation du champ incident
d = linspace(-nbrPoints_z/2, nbrPoints_z/2, nbrPoints_z)
z = linspace(0,fbg1.length, nbrPoints_z)
dz = fbg1.length / nbrPoints_z
Vg = 1.0 / fbg1.beta1(LMBD)
dt = 1e12*(dz / Vg)
dtdz = dureePulse/dt
champ_in = zeros(nbrPoints_z, complex)
pulse = PyOFTK.sechPulse(d,dtdz,0,puissanceCrete,0)
champ_in = PyOFTK.shift(pulse, (nbrPoints_z/2)-4*int(dtdz))


[u_plus, u_moins, u_plus_archive, u_moins_archive, detecteur] = PyOFTK.ossmgpu2(champ_in, fbg1.length, nt, 0.0, fbg1.beta1(LMBD), kapVec, detVec, gamma, 50, 1.0, positionDetecteur)

elapsed = time.time() - t1
print "************ Simulation Parameters ************ "
print "Pulse Energy: " + str(pow(abs(pulse),2).sum()*dt) + " nJ"
print "FBG Length: " + str(fbg1.length) + " m"
print "Temporel iterations number: " + str(nt)
print "Total duration: " + str(nt*dt) + " ps"
print "Kappa: " + str(kapZero) + " [1/m]"
print "Gamma: " + str(gamma) + "m-1 W-1"
print "Detuning: " + str(detZero) + " [1/m]"
print "Detuning: " + str(1000*(LMBD-Longueur_onde_bragg)) + " [nm]"
print "Detuning: " + str(-((2.998e8)*1000*(LMBD-Longueur_onde_bragg))/(pow(LMBD*1000,2))) + " [GHz]"
print "Processing time: " + str(elapsed) + " secondes" + " (" + str(elapsed/60) + " minutes)"
print "********************************************** "

if storeResults == 1:
	PyOFTK.store2hdf5('./temp/z', z)
	PyOFTK.store2hdf5('./temp/kapVec', kapVec)
	PyOFTK.store2hdf5('./temp/detVec', detVec)
	PyOFTK.store2hdf5('./temp/umoins', u_moins_archive)
	PyOFTK.store2hdf5('./temp/uplus', u_plus_archive)

#plot(z,pow(abs(champ_in),2), z,pow(abs(u_plus),2), z,pow(abs(u_moins),2))
#legend(("Champ initiale","u+","u-"))
#grid(True)
#show()

PyOFTK.ossmOutputMP4('bragg_soliton_lin',u_plus_archive.T,u_moins_archive.T, z*1000, kapVec, 20, 0, 0)


