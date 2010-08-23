# *********************************************
#	Test de PyOFTK.ossm()
#	fbg de 9cm
#	20 nm du bandgap
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
from scipy import fftpack
import PyOFTK
import time

t1 = time.time()
storeResults = 0

nbrPoints_z = 40000
# Duree de pulse desire [ps]
dureePulse = 1.0
puissanceCrete = 10000.0
LMBD = 1.030
nt = 40
positionDetecteur = 18500

### Creation d'un fbg ###
fbgStart = 2000
fbgStop = 38000	
# Pas du reseaux
Longueur_onde_bragg = 1.010
Longueur_onde_bragg1 = 1.02
Longueur_onde_bragg2 = 1.03
LBRG = linspace(Longueur_onde_bragg1, Longueur_onde_bragg2, fbgStop-fbgStart)

GrandLambda = Longueur_onde_bragg/(2*1.45)
# Difference d'indice
deltaIndex = 1e-2
fbg1 = PyOFTK.apodizedFBG(2.5, 62.5, 0.05, 0.0, 10e-2, deltaIndex, GrandLambda)
gamma = fbg1.nlGamma(1.03)
detZero = fbg1.detuning(LMBD)
detVec = zeros(nbrPoints_z,double)
detVec[fbgStart:fbgStop] += detZero
#for i in arange(fbgStop-fbgStart):
#	detVec[fbgStart+i] = detZero = fbg1.detuning(LBRG[i])


kapZero = fbg1.kappa(LMBD)
kapVec = zeros(nbrPoints_z,double)
alphaVec = zeros(nbrPoints_z,double)
alphaVec[fbgStart:fbgStop] += -80

# Initialisation du champ incident
d = linspace(-nbrPoints_z/2, nbrPoints_z/2, nbrPoints_z)
z = linspace(0,fbg1.length, nbrPoints_z)
dz = fbg1.length / nbrPoints_z
Vg = 1.0 / fbg1.beta1(LMBD)
dt = 1e12*(dz / Vg)
dtdz = dureePulse/dt
champ_in = zeros(nbrPoints_z, complex)
pulse = PyOFTK.gaussianPulse(d,dtdz,0,puissanceCrete,1,0)
champ_in = PyOFTK.shift(pulse, (nbrPoints_z/2)-4*int(dtdz))


# Kappa apodization profile
#kappaWidth = 1e10
#kapVec += kapZero*exp(-pow(d,4)/kappaWidth)
kapVec[fbgStart:fbgStop] = kapZero

[u_plus, u_moins, u_plus_archive, u_moins_archive, detecteur] = PyOFTK.ossm(champ_in, fbg1.length, nt, alphaVec, fbg1.beta1(LMBD), kapVec, detVec, gamma, 40, 1.0, positionDetecteur)

elapsed = time.time() - t1
print "********* Parametre de la simulation ********* "
print "Energie de l'impulsion en entree: " + str(pow(abs(pulse),2).sum()*dt) + " nJ"
print "Energie de l'impulsion en sortie: " + str(pow(abs(u_plus),2).sum()*dt) + " nJ"
print "Longueur du reseau: " + str(fbg1.length) + " m"
print "Nombre d'iterations temporelles: " + str(nt)
print "Duree totale: " + str(nt*dt) + " ps"
print "Kappa: " + str(kapZero) + " [1/m]"
print "Detuning: " + str(detZero) + " [1/m]"
print "Detuning: " + str(1000*(LMBD-Longueur_onde_bragg)) + " [nm]"
print "Detuning: " + str(((2.998e8)*1000*(LMBD-Longueur_onde_bragg))/(pow(LMBD*1000,2))) + " [GHz]"
print "Processing time: " + str(elapsed) + " secondes" + " (" + str(elapsed/60) + " minutes)"
print "Parametre Beta2 effectif: " + str(fbg1.gBeta2(LMBD)*1e24) + " ps2/m"
print "Parametre Beta3 effectif: " + str(fbg1.gBeta3(LMBD)*1e36) + " ps3/m"
print "********************************************** "

if storeResults == 1:
	PyOFTK.store2hdf5('./temp/z', z)
	PyOFTK.store2hdf5('./temp/kapVec', kapVec)
	PyOFTK.store2hdf5('./temp/detVec', detVec)
	PyOFTK.store2hdf5('./temp/umoins', u_moins_archive)
	PyOFTK.store2hdf5('./temp/uplus', u_plus_archive)

# Spectre
C = 2.99792458e-4
lambdaZero=LMBD*1e-6
T = nt*dt
w = PyOFTK.wspace(T,nt)
vs = fftpack.fftshift(w/(2*pi))
wavelength = (1/((vs/C)+1/(lambdaZero)))*1e9
U_plus = fftpack.fftshift(pow(abs(dt*fftpack.fft(detecteur)/sqrt(2*pi)),2))

# Chirp
t = arange(-T/2, T/2, dt)
phi = arctan2(detecteur.real,detecteur.imag)
nu_inst = -diff(PyOFTK.unfold_arctan2(phi))

plot(z,pow(abs(champ_in),2), z,pow(abs(u_plus),2), z,pow(abs(u_moins),2))
legend(("Champ initiale","u+","u-"))
grid(True)
show()


