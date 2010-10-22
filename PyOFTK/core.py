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
import matplotlib.pyplot as pl
from PyOFTK.utilities import *
import scipy.interpolate as itrp

# Import pygraph
from pygraph.classes.graph import graph
from pygraph.classes.digraph import digraph
from pygraph.algorithms.searching import breadth_first_search
from pygraph.readwrite.dot import write
from pygraph.mixins.basegraph import basegraph


class OFTKDevice:
	''' Class representing a generic PYOFTK device '''

	def __init__(self, description=""):
   		"""
        Initialize an OFTKDevice.
        """
		self.OFTKtype = 0
		self.OFTKdesc = description

	def __repr__(self):
		return str(id(self))

	def _str__(self):
		return self.OFTKdesc

	def info(self):
		print id(self)
		print self.OFTKtype


class OFTKExperiment(digraph, basegraph):
	'''
	Class representing a OFTK Experiment

	An OFTK Experiment is a group of OFTK devices
	'''


	def __init__(self, deviceList=[], autoConnect = False):
		digraph.__init__(self)
		for device in deviceList:
			self.add_node(device)
		if autoConnect:
			for i in arange(len(deviceList)-1):
				self.connect_devices([deviceList[i], deviceList[i+1]])

	def info(self):
		print "This experiment containt " + str(len(self)) + " devices"


	def add_device(self, device):
		'''
		Add a device to the experiment
		'''
		if (type(device) is list):
			self.add_nodes(device)
		else:
			if(isinstance(device, OFTKDevice)):
				self.add_node(device)
			else:
				raise Exception, "Input must be a OFTKDevice"


	def connect_devices(self, deviceList):
		'''
		Connect two devices
		'''
		if len(deviceList)== 2:
			self.add_edge(deviceList)
		else:
			raise Exception, "Only two device can be connected at the same time"


 	def __add__(self,other):
		'''		
		Overloading of the ADD operator
		Merge two experiments in a new instance 'exp'
		'''
		exp = copy.deepcopy(self)
		exp.add_graph(other)	
		return exp

 	def __iadd__(self,other):
		'''		
		Overloading of the iADD operator
		Same as merge_experiments()
		'''
		self.add_graph(other)
	
	def merge_experiments(exp2):
		'''
		Merge two experiments in the current instance
		'''
		self.add_graph(exp2)

	def viz(self, filename):
		'''
		Generate a image of the experiment graph
		'''

		# Construct the image of the graph
		dot = write(self)
		experimentGraphViz = AGraph(string=dot)

		# Parameters of the image
		experimentGraphViz.graph_attr['label']='Experiment'
		experimentGraphViz.graph_attr['dpi'] = '100'
		experimentGraphViz.graph_attr['overlap'] = 'scale'
		experimentGraphViz.node_attr['shape']='box'
		experimentGraphViz.node_attr['label']= ''
		experimentGraphViz.node_attr['color']= 'blue'
		experimentGraphViz.node_attr['style']= 'filled'
		experimentGraphViz.edge_attr['color']='black'
		experimentGraphViz.layout()

		# Write the image in the file
		experimentGraphViz.draw(filename)


class eFieldSVEA:
	'''
	Class representing the SVEA envellope of the field
	'''

	def __init__(self, T, nt, zlength, nz, lambdaZero, ):
		self.nt = nt
		self.t = linspace(-T/2, T/2, nt)
		self.x = zeros(nt, complex)
		self.y = zeros(nt, complex)
		self.w = wspace(T,nt)
		self.vs = fftpack.fftshift(self.w/(2*pi))
		self.currentZ = 0
		dt = T/float(nt)
		C = 2.99792458e-4
		self.wavelength = (1/((self.vs/C)+1/(lambdaZero)))*1e9
		self.xSpectrum = fftpack.fftshift(pow(abs(dt*fftpack.fft(self.x)/sqrt(2*pi)),2))
		self.ySpectrum = fftpack.fftshift(pow(abs(dt*fftpack.fft(self.y)/sqrt(2*pi)),2))

	def __repr__(self):
		return "SVEA Field"

	def pulsePlot(self):
		if self.nt == 1:
			raise Exception, "pulsePlot() function is irrevelant for a CW field"
		else:
			pl.figure(figsize=(8,12))
			pl.subplot(2,1,1)
			pl.plot(self.t, pow(abs(self.x),2),self.t, pow(abs(self.y),2))
			pl.grid(True)
			pl.ylabel("$|u(z,T)|^2$")
			pl.xlabel("$T/T_0$")
			pl.title("Temporal intensity profiles of the field in the fiber")
			pl.legend(("$|E_x|^2$", "$|E_y|^2$"))
			pl.subplot(2,1,2)
			pl.plot(self.wavelength, pow(abs(self.xSpectrum),2),self.wavelength, pow(abs(self.ySpectrum),2))
			pl.grid(True)
			pl.ylabel("Intensity [u.a]")
			pl.xlabel(r'$\lambda$ [nm]')
			pl.title("Intensite spectrum of the field in the fiber")
			pl.legend((r'$\lambda_x$', r'$\lambda_y$'))
			pl.show()


class cwLaser(OFTKDevice):
	'''
	Class representing a cw laser
		* wavelength : Central wavelength of the source [um]
		* power: Mean power of the source [W]
		* linewidth: Linewidth of the source [MHz]
	'''
	
	def __init__(self, wavelength, power, linewidth = 10E6):
		OFTKDevice.__init__(self)
		self.wavelength = wavelength
		self.power = power
		self.linewidth = linewidth

	def __repr__(self):
		return "CW Laser" + self.__instance_name__
	
	def info(self):
		print "Wavelength: " + str(self.wavelength) + " um"
		print "Output power: " + str(self.power) + " mW"


class pumpLaser(cwLaser):
	'''
	Class representing a pump laser
		* wavelength : Central wavelength of the source [um]
		* power: Mean power of the source [W]
		* linewidth: Linewidth of the source [MHz]
	'''
	
	def __init__(self, wavelength, power, linewidth = 10E6):
		cwLaser.__init__(self,wavelength, power, linewidth)

	def __repr__(self):
		return "Pump Laser" + self.__instance_name__
	
	def info(self):
		print "Wavelength: " + str(self.wavelength) + " um"
		print "Output power: " + str(self.power) + " mW"


class pulsedSource(OFTKDevice):
	'''
	Class representing a CW light source
		* wavelength : Wavelength of the source (lambda zero for a pulsed source) [nm]
		* power: Mean power of the source [mW]
		* reprate: Repetition Rate of the source [mhz]
		* pulseShape: Temporal shape of the pulses
	'''
	
	def __init__(self, wavelength, power, reprate):
		OFTKDevice.__init__(self)
		self.wavelength = wavelength
		self.power = power
		self.rapRate = reprate

	def __repr__(self):
		return "Pulsed Source" + self.__instance_name__
	
	def info(self):
		print "Wavelength of this cw source: " + str(self.wavelength) + " um"
		print "Repetition Rate: " + str(self.repRate) + " Mhz"
		print "Output power: " + str(self.power) + " mW"


# Class fibre step-index
class FibreStepIndex(OFTKDevice):
	'''
	Class representing a simple step index fiber
		* fibreCoeur: Radius of the core [um]
		* fibreClad: Radius of the clad [um]
		* Germanium concentration of the core [mass %]
		* Germanium concentration of the clad [mass %]
		* Length of the fiber [m]
	'''

	def __init__(self, fibreCoeur, fibreClad, coeurConcGe, cladConcGe, length, description="Step Index Fiber"):
		OFTKDevice.__init__(self, description)
		self._nbrInput = 1;
		self._nbrOutput = 1;
		self.rayonCoeur = fibreCoeur
		self.rayonClad = fibreClad
		self.coeurConcGe = coeurConcGe
		self.cladConcGe = cladConcGe
		self.length = length
		
	def __str__(self):
		return "Fibre Step Index"

	def __repr__(self):
		return "Fibre Step Index"

	def bgLoss(self, wavelength):
		'''
		Return the background loss for a given wavelength
		Not implemented yet ... 
		'''
		return 0.0

	def vNumber(self, wavelength):
		indiceCoeur = sellmeier(self.coeurConcGe, wavelength)
		indiceClad = sellmeier(self.cladConcGe, wavelength)
		return (2.0*pi/wavelength) * sqrt( pow(indiceCoeur,2.0)-pow(indiceClad,2.0) ) * self.rayonCoeur

	def NA(self, wavelength):
		indiceCoeur = sellmeier(self.coeurConcGe, wavelength)
		indiceClad = sellmeier(self.cladConcGe, wavelength)
		return sqrt( pow(indiceCoeur,2.0)-pow(indiceClad,2.0) )
	
	def width(self, wavelength):
		return self.rayonCoeur*(0.65+(1.619/pow(self.vNumber(wavelength),3.0/2.0))+(2.879/(pow(self.vNumber(wavelength),6.0))))

	def modeOverlap(self, wavelength):
		'''
		Estimation de la largeur du mode pour une
		fibre step-index avec une l'approx gaussienne
			* Wavelength: Wavelength of the mode [um]
		'''
		w = self.width(wavelength)
		return 1 - exp(-pow((self.rayonCoeur/w),2.0))

	def pumpOverlap(self, wavelength):
		return self.modeOverlap(wavelength)

	def coeurSurface(self, wavelength):
		return pi*pow(self.rayonCoeur,2.0)

	def modeArea(self, wavelength):
		return pi*pow(self.width(wavelength),2.0)

	def lpleft(self, u_lpleft, l_lpleft, V_lpleft):
		w = float(sqrt( pow(V_lpleft,2.0)-pow(u_lpleft,2.0) ))
		left_arg = w*sp.kn(l_lpleft-1, w)*sp.jn (l_lpleft, u_lpleft) + u_lpleft*sp.jn(l_lpleft-1, u_lpleft)*sp.kn(l_lpleft, w)
		return abs(left_arg)

	def printLPModes(self, wavelength):
		for l in range(5):
			for m in range(5):
				try:
					print "Mode LP" + str(l)+str(m + 1) + ": u=" + str(self.lpSolve(l,m+1,wavelength,1e-15))
				except:
					print "Mode LP" + str(l)+str(m + 1) + " unsupported"


	def lpSolve2(self, l, m, wavelength, stop_crit):
		'''
		Solve the caracteristic equation with a simple 3 pts method
		return the eigenvalues u with an error stop_crit
		eq. (3.45), Fondamentals of Optical Fiber, John. A. Buck
		
			* l : 		l parameter of the LP mode
			* m : 		m parameter of the LP mode
			* stop_crit:	error (stop criteria)		
		'''

		# Retrieve the vNumber from the class
		vnumber = self.vNumber(wavelength)

		# Find de zeros of the Bessel function		
		zero_j_1 = sp.jn_zeros(l, m)[0]

		# Set the minimum value of the search bound
		if( l==0 ):
			if( m==1 ):
				u_min = 0.0
			else:
				u_min = sp.jn_zeros(0.0, m)[0]
	   	else:
			u_min = sp.jn_zeros(l-1, m)[0]

		# The mode must be supported by the waveguide
		if( u_min < vnumber ):
			err_leaking = 1

		# Set the maximum value of the search bound
		if( vnumber < zero_j_1 ):
			u_max = vnumber
		else:
			u_max = zero_j_1

		# Compute the u value with an error stop_crit 
		# Loop while lpleft(u_guest, l, vnumber) <= stop_crit
		# tite technique de batte a trois points
		u_a = u_min
		u_b = (u_min + u_max)/2.0
		u_error = 1000.0

		while(u_error > stop_crit):
			to_min_a = self.lpleft(u_a, l, vnumber)
			to_min_b = self.lpleft(u_b, l, vnumber)
			diff_b = (to_min_b - to_min_a) / (u_b - u_a)
			u_c = u_b - (to_min_b / diff_b)
	   		
	   		if( u_c <=  u_min ):
	   			u_c = u_min + (vnumber/1000.0)
	   			   		
	   		if( u_c >= u_max ):
	   			u_c = u_max - (vnumber/1000.0)
		
			u_error = abs(u_c - u_b)
			u_a = u_b
			u_b = u_c
	
		return u_a


	def lpSolve(self, l, m, wavelength, stop_crit):
		'''
		Solve the caracteristic equation with fminbound scipy function
		return the eigenvalues u with an error stop_crit
		eq. (3.45), Fonself.taudamentals of Optical Fiber, John. A. Buck
		
			* l : 		l parameter of the LP mode
			* m : 		m parameter of the LP mode
			* stop_crit:	error (stop criteria)		
		'''
		# Retrieve the vNumber from the class
		vnumber = self.vNumber(wavelength)

		# Find de mth zeros of the Bessel_l function		
		zero_j_1 = sp.jn_zeros(l, m).max()

		# Set the minimum value of the search bound
		if( l==0 ):
			if( m==1 ):
				u_min = 0.0
			else:
				u_min = sp.jn_zeros(0.0, m).max()
	   	else:
			u_min = sp.jn_zeros(l-1, m).max()

		# Set the maximum value of the search bound
		if( vnumber < zero_j_1 ):
			u_max = vnumber
		else:
			u_max = zero_j_1

		u_a = op.fminbound(self.lpleft, u_min, u_max, args=(float(l), float(vnumber)), xtol=stop_crit)
	
		return u_a

	
	def betaApprox(self, wavelength):
		indiceClad = sellmeier(self.cladConcGe, wavelength)
		indiceCoeur = sellmeier(self.coeurConcGe, wavelength)
		b = (self.effIndex(wavelength) - indiceClad)/(indiceCoeur - indiceClad)
		delta = (pow(indiceCoeur,2.0) - pow(indiceClad,2.0))/(2.0*pow(indiceCoeur,2.0))
		k_zero = 2.0*pi/(wavelength*1e-6)
		return k_zero*indiceClad*(1.0+b*delta)

	def beta(self, wavelength):
		'''
		Compute Beta [1/m]
		eq. (3.14), Fondamentals of Optical Fiber, John. A. Buck
		'''
		indexCore = sellmeier(self.coeurConcGe, wavelength)
		k_zero = 2.0*pi/(wavelength*1e-6)
		u = self.lpSolve(0.0,1.0,wavelength,1e-15)
		return sqrt(pow(indexCore,2)*pow(k_zero,2)-(pow(u,2)/pow(self.rayonCoeur*1e-6,2)))

	def beta1(self, wavelength):
		'''
		Compute Beta1 [s/m]
		diff(beta)
		'''
		# Size of the diff window [um]
		diffWindow = 0.0001
		# Nbr of points in the window
		diffWindowSize = 5
		wavelenghtMiniArray = linspace(wavelength-diffWindow, wavelength+diffWindow, diffWindowSize)
		deltaLambda = abs(wavelenghtMiniArray[1]-wavelenghtMiniArray[0])
		betaMiniArray = zeros(diffWindowSize,float)

		for i in range(diffWindowSize):
			betaMiniArray[i] = self.beta(wavelenghtMiniArray[i])

		i = 2
		diffOrder4 = (-betaMiniArray[i+2]+8*betaMiniArray[i+1]-8*betaMiniArray[i-1]+betaMiniArray[i-2] )/(12*deltaLambda*1e-6)

		# [beta1 units: s/m]
		return diffOrder4 * (pow(wavelength*1e-6,2)/(-2.0*pi*299792458.0))

	def beta2v(self, wavelength):
		l = len(wavelength)
		beta2Vec = zeros(l, double)
		for i in arange(l):
			beta2Vec[i] = self.beta2(wavelength[i])
		return beta2Vec

	def beta2(self, wavelength):
		'''
		Compute Beta2 [s2/m]
		diffdiff(beta)
		'''
		diffWindow = 0.002
		diffWindowSize = 5
		wavelenghtMiniArray = linspace(wavelength-diffWindow, wavelength+diffWindow, diffWindowSize)
		deltaLambda = abs(wavelenghtMiniArray[1]-wavelenghtMiniArray[0])
		betaMiniArray = zeros(diffWindowSize,float)

		for i in range(diffWindowSize):
			betaMiniArray[i] = self.beta(wavelenghtMiniArray[i])

		i = 2
		diffOrder4 = (-betaMiniArray[i+2]+8*betaMiniArray[i+1]-8*betaMiniArray[i-1]+betaMiniArray[i-2] )/(12*deltaLambda*1e-6)
		diffdiffOrder4 = (-betaMiniArray[i+2]+16*betaMiniArray[i+1]-30*betaMiniArray[i]+16*betaMiniArray[i-1]-betaMiniArray[i-2])/(12*pow(deltaLambda*1e-6,2))

		a = diffdiffOrder4 * pow(-(pow(wavelength*1e-6,2))/(2.0*pi*299792458.0),2)
		b = diffOrder4 * ((pow(wavelength*1e-6,3))/(2.0*pow(pi*299792458.0,2)))

		return (a + b)
	
	def nlGamma(self, wavelength):
		n2Glass = 2.3E-20
		return 2*pi*n2Glass/(wavelength*1e-6*self.modeArea(wavelength)*1e-12)

	def effIndex(self, wavelength):
		indiceClad = sellmeier(self.cladConcGe, wavelength)
		indiceCoeur = sellmeier(self.coeurConcGe, wavelength)
		u = self.lpSolve(0.0,1.0,wavelength,1e-15)
 		return sqrt( pow(indiceCoeur,2.0) - pow((u*self.NA(wavelength)/self.vNumber(wavelength)),2.0) )

	def deltaIndex(self, wavelength):
		indiceClad = sellmeier(self.cladConcGe, wavelength)
		indiceCoeur = sellmeier(self.coeurConcGe, wavelength)
		return (indiceCoeur - indiceClad)

	def totalDispersion(self, wavelength_start, wavelength_end, nbrPoints):
		lambdaArray = linspace(wavelength_start, wavelength_end, nbrPoints)

	def plotLPMode(self,l,m, wavelength, nbrPoints, showCore = 0):
		'''
		Plot the Intensity profile of any lp modes
		eq. (3.74)-(3.75), Fondamentals of Optical Fiber, John. A. Buck
		
			* l : 		l parameter of the LP mode
			* m : 		m parameter of the LP mode
			* nbrPoints:  Plot the mode with a nbrPoints x nbrPoints matrix
			* showCore:	if == 1 Show the core of the fiber
		'''
		self.wlMin = 0.850
		self.wlMax = 1.100
		plotSize = 4*self.rayonCoeur
		I_zero = 1
		IntensiteLP = zeros([nbrPoints, nbrPoints])
		try:
			u = self.lpSolve(l,m,wavelength,1e-15)
			w = sqrt( pow(self.vNumber(wavelength),2.0) - pow(u,2.0) )
		except:
			raise ValueError, "Unsupported mode"


		if l==0:
			for i in range(nbrPoints):
				for j in range(nbrPoints):
					r = float(sqrt(pow(i-nbrPoints/2,2)+pow(j-nbrPoints/2,2)))*(2*plotSize/nbrPoints)
					if (r <= self.rayonCoeur):
						IntensiteLP[i,j] = I_zero*pow(sp.jn(l,(u*r)/self.rayonCoeur),2)
					else:
						IntensiteLP[i,j] = I_zero*pow(sp.jn(l,u)/sp.kn(l,w),2)*pow(sp.kn(l,(w*r)/self.rayonCoeur),2)
		else:
			for i in range(nbrPoints):
				for j in range(nbrPoints):
					try:				
						phi = arctan2(((j-nbrPoints/2)),((i-nbrPoints/2)))
					except:
						phi = pi/2.0
					r = float(sqrt(pow(i-nbrPoints/2,2)+pow(j-nbrPoints/2,2)))*(2*plotSize/nbrPoints)
					if (r <= self.rayonCoeur):
						IntensiteLP[i,j] = I_zero*pow(sp.jn(l,(u*r)/self.rayonCoeur),2)*pow(sin(l*phi),2)
					else:
						IntensiteLP[i,j] = I_zero*pow(sp.jn(l,u)/sp.kn(l,w),2)*pow(sp.kn(l,(w*r)/self.rayonCoeur),2)*pow(sin(l*phi),2)



		a = pl.subplot(111, aspect='equal')

		pl.imshow(IntensiteLP, aspect='equal', cmap=pl.cm.gray, zorder=0, interpolation='nearest')
		
		
		if showCore:
			# if showCore is set to 1 the core of the fiber is draw over the mode
			ells = Ellipse((nbrPoints/2,nbrPoints/2), nbrPoints/4, nbrPoints/4)
			ells.set_clip_box(a.bbox)
			ells.set_alpha(0.5)
			ells.set_edgecolor([0.25,0.25,0.25])
			ells.set_facecolor('none')
			a.add_artist(ells)

			pl.xlim(0, nbrPoints-1)
			pl.ylim(0, nbrPoints-1)

		pl.show()


class Fiber():
	'''
	Class representing am axisymmetric optical fiber with
	an arbitrary index profile
		* indexProfile: Fiber radial index profile		
		* length:		Length of the fiber
	'''
	def __init__(self, indexProfile, length):
		self.length = length

	def __repr__(self):
		return "Axisymmetric optical fiber"


class coupler1x2(FibreStepIndex):
	def __init__(self, fibreCoeur, fibreClad, coeurConcGe, cladConcGe, length, N, description="1x2 Coupler"):
		FibreStepIndex.__init__(self, fibreCoeur, fibreClad, coeurConcGe, cladConcGe, length, description)


class YbDopedFiber(FibreStepIndex):
	'''
	Class representing a typical Ytterbium-doped step index fiber
	Inherihit from FibreStepIndex class
	'''
	
	def __init__(self, fibreCoeur, fibreClad, coeurConcGe, cladConcGe, length, N, description="Yb-Doped Step Index Fiber"):
		FibreStepIndex.__init__(self, fibreCoeur, fibreClad, coeurConcGe, cladConcGe, length, description)
		self.concDopant = N
		self.tau = 0.850E-3
		self.wlMin = 0.850
		self.wlMax = 1.100
		csFile = load('cross_section_ytterbium.npz')
		self.csEm = csFile['cross_em'][:,1]
		self.csAbs = csFile['cross_abs'][:,1]
		self.csAbsZoom = csFile['cross_abs_zoom'][:,1]
		self.csEmWL = csFile['cross_em'][:,0]
		self.csAbsWL = csFile['cross_abs'][:,0]
		self.csAbsZoomWL = csFile['cross_abs_zoom'][:,0]
		self.csEmSpline = itrp.splrep(self.csEmWL, self.csEm)
		self.csAbsSpline = itrp.splrep(self.csAbsWL, self.csAbs)
		self.csAbsZoomSpline = itrp.splrep(self.csAbsZoomWL, self.csAbsZoom)

	def __repr__(self):
		return "Yb-doped Step Index Fiber"

	def set_tau(self,tau):
		self.tau = tau

	def crossSection(self, WLum):
		'''
		Return the emission and absorption cross-section for any
		wavelength between 850 and 1150 nm
		'''
		WL = WLum*1E3
		if isinstance(WLum, numpy.ndarray):
			LNGTH = len(WL)
			csEm = zeros(LNGTH)
			csAbs = zeros(LNGTH)
			for i in arange(LNGTH):
				if WL[i] < self.csAbsZoomWL.min():
					csEm[i] = itrp.splev(WL[i],self.csEmSpline)
					csAbs[i] = itrp.splev(WL[i],self.csAbsSpline)
				else:
					csEm[i] = itrp.splev(WL[i],self.csEmSpline)
					csAbs[i] = itrp.splev(WL[i],self.csAbsZoomSpline)
			return [csEm,csAbs]
		else:
			if WL < self.csAbsZoomWL.min():
				return [itrp.splev(WL,self.csEmSpline),itrp.splev(WL,self.csAbsSpline)]
			else:
				return [itrp.splev(WL,self.csEmSpline),itrp.splev(WL,self.csAbsZoomSpline)]


class YbDopedDCOF(YbDopedFiber):
	'''
	Class representing a typical double-clad Ytterbium-doped step index fiber
	Inherihit from YbDopedFiber class
	'''
	
	def __init__(self, fibreCoeur, fibreClad, coeurConcGe, cladConcGe, length, N, description="Yb-Doped Double-clad Step Index Fiber"):
		YbDopedFiber.__init__(self, fibreCoeur, fibreClad, coeurConcGe, cladConcGe, length, N, description)

	def __repr__(self):
		return "Yb-doped double-clad Step Index Fiber"

	def pumpOverlap(self, wavelength):
		return pow(self.rayonCoeur,2.0) / pow(self.rayonClad,2)


class ErDopedFiber(FibreStepIndex):
	'''
	Class representing a typical Erbium-doped step index fiber
	Inherihit from FibreStepIndex class
	'''
	
	def __init__(self, fibreCoeur, fibreClad, coeurConcGe, cladConcGe, length, N, description="Er-Doped Step Index Fiber"):
		FibreStepIndex.__init__(self, fibreCoeur, fibreClad, coeurConcGe, cladConcGe, length, description)
		self.concDopant = N
		self.tau = 10E-3
		self.wlMin = 1.450
		self.wlMax = 1.650
		csFile = load('cross_section_erbium.npz')
		self.csEm = csFile['cross_em']
		self.csAbs = csFile['cross_abs']
		self.csAbs980 = csFile['cross_abs_980']
		self.csWL = csFile['wavelength']
		self.csWL980 = csFile['wavelength980']
		self.csEmSpline = itrp.splrep(self.csWL, self.csEm)
		self.csAbsSpline = itrp.splrep(self.csWL, self.csAbs)
		self.csAbs980Spline = itrp.splrep(self.csWL980, self.csAbs980)
	def __repr__(self):
		return "Er-doped Step Index Fiber"

	def crossSection(self, WLum):
		'''
		Return the emission and absorption cross-section for any
		wavelength between 1450 and 1650 nm
		'''
		WL = WLum*1E3
		if isinstance(WLum, numpy.ndarray):
			LNGTH = len(WL)
			csEm = zeros(LNGTH)
			csAbs = zeros(LNGTH)
			for i in arange(LNGTH):
				if WL[i] < self.csWL980.max() and WL[i] > self.csWL980.min():
					csEm[i] = 0.0
					csAbs[i] = itrp.splev(WL[i],self.csAbs980Spline)
				else:
					csEm[i] = itrp.splev(WL[i],self.csEmSpline)
					csAbs[i] = itrp.splev(WL[i],self.csAbsSpline)
			return [csEm,csAbs]
		else:
			if WL < self.csWL980.max() and WL > self.csWL980.min():
				return [0.0, itrp.splev(WL,self.csAbs980Spline)]
			else:
				return [itrp.splev(WL,self.csEmSpline),itrp.splev(WL,self.csAbsSpline)]


# Definition de la class fbg heritee de la class FibreStepIndex
class fbg(FibreStepIndex):
	''' 
	Class representing a fiber bragg grating in a step index fiber
	Inherihit from FibreStepIndex class
	'''

	def __init__(self, fibreCoeur, fibreClad, coeurConcGe, cladConcGe, L, fbgDelta, fbgLambda):
		self.rayonCoeur = fibreCoeur
		self.rayonClad = fibreClad
		self.coeurConcGe = coeurConcGe
		self.cladConcGe = cladConcGe

		self.deltaIndex = fbgDelta
		self.averageIndex = 1.45
		self.braggWavelength = 2*self.averageIndex*fbgLambda
		self.angFrequencyBragg = (pi*299792458)/(self.averageIndex*fbgLambda*1e-6)
		self.braggWaveNumber = pi / fbgLambda

	def kappa(self, wavelength):
		return 2.0*pi*self.deltaIndex / (wavelength*1e-6)

	def detuning(self, wavelength):
		angFrequency = 2*pi*(299792458)/(wavelength*1e-6)
		return (self.averageIndex/299792458)*(angFrequency-self.angFrequencyBragg)


class simpleFBG(FibreStepIndex):
	'''
	Class representing a uniform fiber bragg grating in a step index fiber
	Inherihit from FibreStepIndex class
	'''

	def __init__(self, fibreCoeur, fibreClad, coeurConcGe, cladConcGe, L, fbgDelta, fbgLambda):
		self.rayonCoeur = fibreCoeur
		self.rayonClad = fibreClad
		self.coeurConcGe = coeurConcGe
		self.cladConcGe = cladConcGe

		self.deltaIndex = fbgDelta
		self.averageIndex = 1.45
		self.braggWavelength = 2*self.averageIndex*fbgLambda
		self.angFrequencyBragg = (pi*299792458)/(self.averageIndex*fbgLambda*1e-6)
		self.braggWaveNumber = pi / fbgLambda

	def kappa(self, wavelength):
		return 2.0*pi*self.deltaIndex / (wavelength*1e-6)

	def detuning(self, wavelength):
		angFrequency = 2*pi*(299792458)/(wavelength*1e-6)
		return (self.averageIndex/299792458)*(angFrequency-self.angFrequencyBragg)

	def gBeta2(self, wavelength):
		a = sign(self.detuning(wavelength))*pow(self.kappa(wavelength),2)*pow(self.beta1(wavelength),2)
		b = pow((pow(self.detuning(wavelength),2) - pow(self.kappa(wavelength),2)),3.0/2.0)
		return -a/b

	def gBeta3(self, wavelength):
		a = sign(self.detuning(wavelength))*pow(self.kappa(wavelength),2)*pow(self.beta1(wavelength),2)
		b = pow((pow(self.detuning(wavelength),2) - pow(self.kappa(wavelength),2)),3.0/2.0)
		return -a/b


class apodizedFBG(FibreStepIndex):
	'''
	Class representing a apodized fiber bragg grating in a step index fiber
	Inherihit from FibreStepIndex class
	'''

	def __init__(self, fibreCoeur, fibreClad, coeurConcGe, cladConcGe, L, fbgDelta, fbgLambda):
		self.rayonCoeur = fibreCoeur
		self.rayonClad = fibreClad
		self.coeurConcGe = coeurConcGe
		self.cladConcGe = cladConcGe
		self.length = L

		self.deltaIndex = fbgDelta
		self.eta = 0.8
		self.averageIndex = 1.45
		self.braggWavelength = 2*self.averageIndex*fbgLambda
		self.angFrequencyBragg = (pi*299792458)/(self.averageIndex*fbgLambda*1e-6)
		self.braggWaveNumber = pi / fbgLambda


	def kappa(self, wavelength):
		# Coupling coefficient [m-1]

		if isinstance(wavelength, numpy.ndarray):
			LNGTH = len(wavelength)
			kapVec = zeros(LNGTH)
			for i in arange(LNGTH):
				kapVec[i] = pi*self.deltaIndex*self.eta / (wavelength[i]*1e-6)
			return kapVec
		else:
			return pi*self.deltaIndex*self.eta / (wavelength*1e-6)

	def detuning(self, wavelength):
		if isinstance(wavelength, numpy.ndarray):
			LNGTH = len(wavelength)
			detVec = zeros(LNGTH)
			for i in arange(LNGTH):
				angFrequency = 2*pi*(299792458)/(wavelength[i]*1e-6)
				detVec[i] = (self.averageIndex/299792458)*(angFrequency-self.angFrequencyBragg)
			return detVec
		else:
			angFrequency = 2*pi*(299792458)/(wavelength*1e-6)
			return (self.averageIndex/299792458)*(angFrequency-self.angFrequencyBragg)

	def detuningw(self, angFrequency):
		return (self.averageIndex/299792458)*(angFrequency-self.angFrequencyBragg)

	def nu(self, wavelength):
		return sqrt(1 - pow(self.kappa(wavelength),2)/pow(self.detuning(wavelength),2))

 	def gGamma(self, wavelength):
		return self.nlGamma(wavelength) * ((3-pow(self.nu(wavelength),2))/(2*self.nu(wavelength)))

	def q(self, wavelength):
		'''
		Output q for a given wavelength
		[1/m]
		'''
		if isinstance(wavelength, numpy.ndarray):
			LNGTH = len(wavelength)
			qVec = zeros(LNGTH)
			for i in arange(LNGTH):
				qVec[i] = sqrt(pow(self.detuning(wavelength[i]),2) - pow(self.kappa(wavelength[i]),2))
			return qVec
		else:
			q = sqrt(pow(self.detuning(wavelength),2) - pow(self.kappa(wavelength),2))
			sign = 1
			'''
			if abs((q-self.detuning(wavelength))/self.kappa(wavelength)) < 1:
				sign = 1
			else:
				sign = -1
			'''
			return sign*q

	def qdet(self, det):
		'''
		Output q for a given detuning
		[1/m]
		'''
		if isinstance(det, numpy.ndarray):
			LNGTH = len(det)
			qVec = zeros(LNGTH)
			for i in arange(LNGTH):
				qVec[i] = sqrt(pow(det[i],2) - pow(self.kappa(det[i]),2))
			return qVec
		else:
			return sqrt(pow(det,2) - pow(self.kappa(det),2))

	def qw(self, angFrequency):
		'''
		Output q for a given angular frequency
		[1/m]
		'''
		if isinstance(angFrequency, numpy.ndarray):
			LNGTH = len(angFrequency)
			qVec = zeros(LNGTH, float)
			for i in arange(LNGTH):
				qVec[i] = sqrt(pow(self.detuningw(angFrequency[i]),2) - pow(self.kappa(angFrequency[i]),2))
			return qVec
		else:
			return sqrt(pow(self.detuningw(angFrequency),2) - pow(self.kappa(angFrequency),2))

	def reflectivity(self, wavelength):
		# Equation 1.3.29 d'Agrawal (Application)
		if isinstance(wavelength, numpy.ndarray):
			LNGTH = len(wavelength)
			wlArray = zeros(LNGTH)
			for i in arange(LNGTH):
				a = 1.0j * self.kappa(wavelength[i]) * sin(self.q(wavelength[i])*self.length)
				b = self.q(wavelength[i]) * cos(self.q(wavelength[i])*self.length)
				c = -1.0j * self.detuning(wavelength[i]) * sin(self.q(wavelength[i])*self.length)
				wlArray[i] = pow(abs(a/(b+c)),2)
			return wlArray
		else:
			a = 1.0j * self.kappa(wavelength) * sin(self.q(wavelength)*self.length)
			b = self.q(wavelength) * cos(self.q(wavelength)*self.length)
			c = -1.0j * self.detuning(wavelength) * sin(self.q(wavelength)*self.length)
			return pow(abs(a/(b+c)),2)

	def maxReflectivity(self):
		
		return pow(tanh(self.kappa(self.braggWavelength)*self.length),2)

	def gBeta2(self, wavelength):
		# Grating dispersion [s2 m-1]
		a = pow((self.averageIndex/299792458),2)
		b = 1.0/self.detuning(wavelength)
		c = pow(self.kappa(wavelength),2)/pow(self.detuning(wavelength),2)
		d = pow(1-pow(self.kappa(wavelength)/self.detuning(wavelength),2),3.0/2.0)
		return -(a*b*c)/d

	def gBeta3(self, wavelength):
		# Grating dispersion [s3 m-1]
		a = pow((self.averageIndex/299792458),3)
		b = 1.0/pow(self.detuning(wavelength),2)
		c = pow(self.kappa(wavelength),2)/pow(self.detuning(wavelength),2)
		d = pow(1-pow(self.kappa(wavelength)/self.detuning(wavelength),2),5.0/2.0)
		return 3*(a*b*c)/d

	def gBetaMwl(self, lambdaVec, M):
		# Grating dispersion order M [sM m-1]
		# Unstable at high M!
		qVec = self.q(lambdaVec)
		
		[diffM, lambdaVecCrop]= diffOrder4N(qVec, lambdaVec, M)
		
		nbrSamples = len(diffM)

		# Erreur: je dois modifier le facteur de conversion lambda...
		return diffM[nbrSamples/2]*(-pow(lambdaVec[nbrSamples/2],2)/(2*pi))

	def gBetaM(self, angFreqVec, M):
		# Grating dispersion order M [sM m-1]
		# Unstable at high M!
		qVec = self.qw(angFreqVec)
		
		[diffM, angFreqVecCrop]= diffOrder4N(qVec, angFreqVec, M)
		
		nbrSamples = len(diffM)

		return diffM[nbrSamples/2]


	def gBetaMApprox(self, wavelength, M):
		'''		
		Grating dispersion order M [s^M m-1]
		Approx. using eq. 1.3.24 of Agrawal Appplications
		'''
		nbrSamples = 1000
		windowsRatio = 0.1
		detZero = self.detuning(wavelength)
		detVec = linspace(detZero-(detZero*windowsRatio), detZero+(detZero*windowsRatio), nbrSamples)
		deltaDet = detVec[1] - detVec[0]
		qVec = zeros(nbrSamples)

		qVec = self.qdet(detVec)
		
		diffOrder4 = zeros(nbrSamples, float)
		for m in arange(M)+1:
			# Diff
			for i in arange(2*m, nbrSamples-2*m):
				diffOrder4[i] = (-qVec[i+2]+8*qVec[i+1]-8*qVec[i-1]+qVec[i-2] )/(12*deltaDet)
			qVec = diffOrder4

		return pow(self.beta1(wavelength),M)*diffOrder4[nbrSamples/2]

	def vg(self, wavelength):
		'''		
		Speed of light in fiber without the effect of the grating
		'''
		return (299792458/self.averageIndex)*sqrt(1-(pow(self.kappa(wavelength),2)/pow(self.detuning(wavelength),2)) )

	def tmMethod(self):
		'''
		Implementation of the Transfer Matrix Method
		'''
		

class apodizedFBGv(FibreStepIndex):

	def __init__(self, fibreCoeur, fibreClad, coeurConcGe, cladConcGe, L, fbgDelta, fbgLambda, indexProfile):
		self.rayonCoeur = fibreCoeur
		self.rayonClad = fibreClad
		self.coeurConcGe = coeurConcGe
		self.cladConcGe = cladConcGe
		self.length = L
		self.deltaIndex = fbgDelta
		self.eta = 0.8
		self.averageIndex = 1.45
		# Vector
		self.indexProfile = indexProfile
		self.braggWavelength = 2*self.indexProfile*fbgLambda
		self.angFrequencyBragg = (pi*299792458)/(self.indexProfile*fbgLambda*1e-6)
		self.braggWaveNumber = pi / fbgLambda

	def detuning(self, wavelength):
		angFrequency = 2*pi*(299792458)/(wavelength*1e-6)
		return (self.indexProfile/299792458)*(angFrequency-self.angFrequencyBragg)

	def kappa(self, wavelength):
		# Coupling coefficient [m-1]
		return pi*self.deltaIndex*self.eta / (self.braggWavelength*1e-6)
	

