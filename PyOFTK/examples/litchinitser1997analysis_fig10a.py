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

nbrPoints_z = pow(2,16)
# Duree de pulse desire [ps]
dureePulse = 40.0
puissanceCrete = 1.0
LMBD = 1.55000
nt = 70000
positionDetecteur = 30100

### Creation d'un fbg ###
fbgStart = 20000
fbgStop = 33107	
# Pas du reseaux
Longueur_onde_bragg = 1.55005
Longueur_onde_bragg1 = 1.02
Longueur_onde_bragg2 = 1.03
LBRG = linspace(Longueur_onde_bragg1, Longueur_onde_bragg2, fbgStop-fbgStart)

GrandLambda = Longueur_onde_bragg/(2*1.45)
# Difference d'indice
deltaIndex = 1.931e-5
fbg1 = PyOFTK.apodizedFBG(2.5, 62.5, 0.05, 0.0, 50e-2, deltaIndex, GrandLambda)
gamma = fbg1.nlGamma(1.03)
detZero = fbg1.detuning(LMBD)
detVec = zeros(nbrPoints_z,double)
detVec[fbgStart:fbgStop] += detZero
#for i in arange(fbgStop-fbgStart):
#	detVec[fbgStart+i] = detZero = fbg1.detuning(LBRG[i])


kapZero = fbg1.kappa(LMBD)
kapVec = zeros(nbrPoints_z,double)
alphaVec = zeros(nbrPoints_z,double)
alphaVec[fbgStart:fbgStop] += 0.0

# Initialisation du champ incident
d = linspace(-nbrPoints_z/2, nbrPoints_z/2, nbrPoints_z)
z = linspace(0,fbg1.length, nbrPoints_z)
dz = fbg1.length / nbrPoints_z
Vg = 1.0 / fbg1.beta1(LMBD)
dt = 1e12*(dz / Vg)
dtdz = dureePulse/dt
champ_in = zeros(nbrPoints_z, complex)
pulse = PyOFTK.gaussianPulse(d,dtdz,0,puissanceCrete,1,0)


# Dispersion de 100 km 

try:
	 u_out_x1 = PyOFTK.hdf5load("100km1.h5")
except:
	T = nbrPoints_z * dt
	w = PyOFTK.wspace(T,nt)
	vs = fftpack.fftshift(w/(2*pi))
	C = 2.99792458e-4
	lambdaZero = LMBD*1.0e-6
	wavelength = (1/((vs/C)+1/(lambdaZero)))*1e9
	betap1 = array([0.0, 0.0, -0.020, 0.0])
	gainBandwidth = 400
	alpha = array([-0.0])
	alphaSatPeak = alpha[0]
	alphaSat = PyOFTK.gainProfil(wavelength, alphaSatPeak, gainBandwidth)
	#gamma = 0.0053675371798866684
	gamma = 0.0
	dz_100km = 10
	nz_100km = 10000
	u_ini_y = zeros(nbrPoints_z,complex)
	[u_out_x1, u_out_y1, outputParam] = PyOFTK.vector(pulse, u_ini_y, dt, dz_100km, nz_100km, fftpack.fftshift(alphaSat), fftpack.fftshift(alphaSat), betap1, betap1, gamma, 0, 50, 1e-5, 0, 0)

champ_in = PyOFTK.shift(u_out_x1, (nbrPoints_z/2)-10*int(dtdz))

# Kappa apodization profile
#kappaWidth = 1e10
#kapVec += kapZero*exp(-pow(d,4)/kappaWidth)
kapVec[fbgStart:fbgStop] = kapZero

[u_plus, u_moins, u_plus_archive, u_moins_archive, detecteur] = PyOFTK.ossm(champ_in, fbg1.length, nt, alphaVec, fbg1.beta1(LMBD), kapVec, detVec, gamma, 140, 1.0, positionDetecteur)

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


