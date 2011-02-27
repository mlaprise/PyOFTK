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
from vtk import *
from tables import *
from matplotlib.patches import Ellipse
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.signal as signal
import scipy.special as sp
import scipy.optimize as op
import scipy.fftpack as fftpack
from numpy.fft import fftshift
import numpy.linalg as la
import gtk, gobject
import os.path
import copy
import time
import sys
import pyfits
import numpy
from scipy.optimize import leastsq
import pulse


pi = 3.14159265358979323846264338327950288419716939937510582097494459


def chi2(array1, array2):
	'''
	Evaluate the error between two arrays with the chi2
	'''
	nbrPoints = shape(array1)[0]
	return sqrt( pow((array2-array1),2).sum() ) / nbrPoints


def computePhaseSpace(t, w, temporalProfile, spectrumProfile, nbPass):

	tempWidth = zeros(nbPass)
	specWidth = zeros(nbPass)
	for i in arange(nbPass):
		tempWidth[i] = rmsWidth(t, temporalProfile[i])
		specWidth[i] = rmsWidth(w, spectrumProfile[i])

	return [tempWidth, specWidth]


def diffOrder4(y, x):
	dx = x[1]-x[0]
	nbrSamples = len(y)
	diff = zeros(nbrSamples, float)

	for i in arange(2, nbrSamples-2):
		diff[i] = (-y[i+2]+8*y[i+1]-8*y[i-1]+y[i-2] )/(12*dx)
	
	return diff[2:nbrSamples-2]

def diffOrder4N(y, x, N):
	nbrSamples = len(y)
	diffN = zeros(nbrSamples)
	yBuffer = y
	xBuffer = x
	for n in arange(N)+1:
		diffN = diffOrder4(yBuffer,xBuffer)
		yBuffer = diffN
		xBuffer = xBuffer[2:nbrSamples-2]
		nbrSamples -= 4
	
	return [yBuffer, xBuffer]


def TBWP(t, SVEAAmp):
	'''
	Time-bandwidth product (TBWP)
	!!!!!!!!!!!! Not finish !!!!!!!!!!!!!
	'''
	w = angFreq(t)
	vs = fftpack.fftshift(w/(2*pi))
	intensity = pow(abs(SVEAAmp),2)
	timeWidth = FWHM(t, pow(abs(SVEAAmp),2))
	pulseBW = FWHM(vs, pow(abs(vs),2))
	return timeWidth*pulseBW


def pulseSpectrum(t, SVEAAmp, lambdaZero = 0.0, units = 'nm'):
	'''
	Compute the spectrum of o SVEA pulse center at lambdaZero
		* t: time vector
		* SVEAAmp: SVEA enveloppe of the pulse
		* lambdaZero: center of the pulse [m]
		* units: Units of the output ['nm','um','m']
	'''

	C = 2.99792458e-4
	nt = len(t)
	dt = t[1] - t[0]
	T =  t.max()-t.min()
	w = wspace(T,nt)
	vs = fftpack.fftshift(w/(2*pi))

	# Assign uniScale 
	unitScale = {
	  'nm': lambda: 1.0e9,
	  'um': lambda: 1.0e6,
	  'm': lambda: 1.0
	}[units]()

	if lambdaZero != 0.0:
		wavelength = ( 1.0/( (vs/C)+1.0/(lambdaZero) ) )*unitScale
		return [wavelength, fftpack.fftshift(pow(abs(dt*fftpack.fft(SVEAAmp)/sqrt(2.0*pi)),2))]
	else:
		return [vs, fftpack.fftshift(pow(abs(dt*fftpack.fft(SVEAAmp)/sqrt(2.0*pi)),2))]


def satAbsorberSchreiber(amplitude, R_unsat, R_sat, P_sat):
	intensity = pow(abs(amplitude),2)
	transmittance = R_unsat + R_sat*(1-(1.0/(1+(intensity/P_sat))))
	return amplitude*transmittance


def satAbsorber(amplitude, intSat, depthMod):
	intensity = pow(abs(amplitude),2)
	transmittance = 1-(depthMod/(1+(intensity/intSat)))
	return amplitude*transmittance


def bandPassFilter(t, SVEAAmp, lambdaZero, filterBW, apodisation = 1.0):
	'''
	Bandpass Filter
	'''
	C = 2.99792458e-4
	T = t.max()-t.min()
	nt = len(t)
	dt = t[1]-t[0]
	w = wspace(T,nt)
	vs = fftshift(w/(2*pi))
	wavelength = (1/((vs/C)+1/(lambdaZero*1E-9)))*1e9

	spectrumAmp = fftshift(fftpack.fft(SVEAAmp))

	filterEnv=pulse.gaussianPulse(wavelength, filterBW, lambdaZero, 1.0, apodisation)

	return [wavelength, fftpack.ifft(spectrumAmp*filterEnv)]


def slowSatAbsorber(amplitude, recoveryTime, energySat, depthMod):
	intensity = pow(abs(amplitude),2)
	systemeEquation = lambda y,t : array([-y[0]/recoveryTime + depthMod/recoveryTime - y[0]*intensity(t)/energySat])
	y0=array([depMod+0.1])
	y=integrate.odeint(systemeEquation,y0,time)
	return amplitude*y


def labGainProfil(wavelength, alphaSatPeak, gainBandwidth):
	'''	
	Generate a gain profile from experimentales cross-section values
	'''
	# Construction d'une curve de cross-Section interpole
	eCrossSection = load('CrossSectionEm.dat')
	aCrossSection = load('CrossSectionAbs.dat')
	aCrossSectionZoom = load('CrossSectionAbs_Zoom.dat')

	# Petite passe pour que le pics a 975 fit en emission et en absorption
	aCrossSection_norm = aCrossSection[:,1]*(eCrossSection[:,1].max()/aCrossSection[:,1].max())

	eCS_spline = scipy.interpolate.splrep(eCrossSection[:,0],  signal.wiener(eCrossSection[:,1],5,10))
	aCS_spline = scipy.interpolate.splrep(aCrossSection[:,0]*1E9,  signal.wiener(aCrossSection_norm,5,10))
	aCSZoom_spline = scipy.interpolate.splrep(aCrossSectionZoom[:,0]*1E9,  signal.wiener(aCrossSectionZoom[:,1],5,10))

	eCS_WL = arange(850, 1100, 1)
	aCS_WL = arange(850, 1001 ,1)
	aCSZoom_WL = arange(1001, 1100, 1)
	eCS_int = scipy.interpolate.splev(eCS_WL, eCS_spline)
	aCSLeft_int = scipy.interpolate.splev(aCS_WL, aCS_spline)
	aCSRight_int = scipy.interpolate.splev(aCSZoom_WL, aCSZoom_spline)
	aCS_int = r_[aCSLeft_int, aCSRight_int]


def gainProfil(wavelength, alphaSatPeak, gainBandwidth):
	size = len(wavelength)
	lorentzAmplitude = 1/(pi)
	return (((alphaSatPeak/lorentzAmplitude)*(1/pi))/(1+pow((wavelength-wavelength[size/2])/gainBandwidth,2)))


def modeWidthGaussApprox(a, V):
	return a*(0.65+(1.619/pow(V, (3.0/2.0)))+(2.879/pow(V,6)))


def stepIndexNA(coreIndex, cladIndex):
	return sqrt( pow(coreindex,2) - pow(cladindex,2) )


def stepIndexVnumber(a, coreIndex, cladIndex, wavelength):
	NA = stepIndexNA(coreIndex, cladIndex)
	return 2*pi*NA*a/wavelength;


def satAbsorber(amplitude, intSat, depthMod):
	intensity = pow(abs(amplitude),2)
	transmittance = 1-(depthMod/(1+(intensity/intSat)))
	return amplitude*transmittance


def slowSatAbsorber(amplitude, recoveryTime, energySat, depthMod):
	intensity = pow(abs(amplitude),2)
	systemeEquation = lambda y,t : array([-y[0]/recoveryTime + depthMod/recoveryTime - y[0]*intensity(t)/energySat])
	y0=array([depMod+0.1])
	y=integrate.odeint(systemeEquation,y0,time)
	return amplitude*y


def angFreq(t, lambdaZero = 0.0, nt = 1):
	'''
	Constructs a linearly-spaced vector of angular frequencies
	t: Linearly space time array or total time length
	lambdaZero: Center wavelength [m]
	'''
	C = 2.998E8
	if isinstance(t, numpy.ndarray):
		nt = len(t)
		dt = t[1] - t[0]
	else:
		dt = t/float(nt)

	if lambdaZero == 0.0:
		w = 2.0*pi*concatenate((arange(0,nt/2.0),arange(-nt/2.0, 0))) / (dt*nt)
	else:
		w = 2.0*pi*concatenate((arange(0,nt/2.0),arange(-nt/2.0, 0))) / (dt*nt) + (2*pi*C/lambdaZero)
	return fftpack.fftshift(w)


def wspace(T,nt):
	'''
	Constructs a linearly-spaced vector of angular frequencies
	'''
	dt = T/float(nt)
	w = 2*pi*arange(0,nt)/T
	kv = where(w >= pi/dt)
	w[kv] = w[kv] - 2*pi/dt
	return w



def sellmeier(concGe, wavelenght):
	'''
	Simple function for the computing of the sellmeier coefficient of a ge-doped glass

		* wavelenght : Wavelenght vector in microns (the coeff. are computed at this wavelenght)
 		* concGe : Germanium concentration of the glass
	'''
	AS1 = 0.69616630
	AS2 = 0.40794260
	AS3 = 0.89747940
	lS1 = 0.068404300
	lS2 = 0.11624140
	lS3 = 9.8961610
   
	AG1 = 0.80686642
	AG2 = 0.71815848
	AG3 = 0.85416831
	lG1 = 0.068972606
	lG2 = 0.15396605
	lG3 = 11.841931

	terme1 = ((AS1+concGe*(AG1-AS1))*pow(wavelenght,2))/(pow(wavelenght,2) -  pow( (lS1+concGe*(lG1-lS1)),2))
	terme2 = ((AS2+concGe*(AG2-AS2))*pow(wavelenght,2))/(pow(wavelenght,2) -  pow( (lS2+concGe*(lG2-lS2)),2))
	terme3 = ((AS3+concGe*(AG3-AS3))*pow(wavelenght,2))/(pow(wavelenght,2) -  pow( (lS3+concGe*(lG3-lS3)),2))
	
	return sqrt( 1 + terme1 + terme2 + terme3 )


def unfold_arctan2(arctan2_output, phiZero = 0):

	size = len(arctan2_output)

	if phiZero == 0:
		phiZero = size/2

	unfold_arctan2_output = copy.deepcopy(arctan2_output)
	derive = diff(arctan2_output)
	seuil = derive.max()/4
	disc = where( (diff(arctan2_output) > seuil) | (diff(arctan2_output) < -seuil))
	discNbr = disc[0].shape[0]
	deriveSign = sign(derive[phiZero + 1])

	for i in arange(phiZero,size):
		discBefore = len(where( (disc[0] < i) & (disc[0] > (phiZero)))[0])
		unfold_arctan2_output[i] = arctan2_output[i] + (2*pi*discBefore*deriveSign)

	for i in arange(0,phiZero):
		discBefore = len(where( (disc[0] >= i) & (disc[0] < (phiZero)))[0])
		unfold_arctan2_output[i] = arctan2_output[i] + (2*pi*discBefore*deriveSign)

	return unfold_arctan2_output


def unfold_arctan2b(arctan2_output):

	size = len(arctan2_output)

	unfold_arctan2_output = copy.deepcopy(arctan2_output)
	derive = diff(arctan2_output)
	seuil = derive.max()/4
	disc = where( (diff(arctan2_output) > seuil) | (diff(arctan2_output) < -seuil))
	discNbr = disc[0].shape[0]
	deriveSign = sign(derive[(size/2) + 1])

	for i in arange(size/2,size):
		discBefore = len(where( (disc[0] < i) & (disc[0] > (size/2)))[0])
		unfold_arctan2_output[i] = arctan2_output[i] + (2*pi*discBefore*deriveSign)

	for i in arange(0,size/2):
		discBefore = len(where( (disc[0] >= i) & (disc[0] < (size/2)))[0])
		unfold_arctan2_output[i] = arctan2_output[i] + (2*pi*discBefore*deriveSign)

	return unfold_arctan2_output


def nuInst(complexAmp, phiZero = 0):
	'''
	Compute the instantaneous frequency of a SVEA Field
	'''
	phi_out = arctan2(complexAmp.real,complexAmp.imag)
	return -diff(unfold_arctan2(phi_out, phiZero))


def phase(complexAmp):
	return arctan2(complexAmp.real,complexAmp.imag)


def constructLambda(lambdaZero, bandWidth, nb_points):
	 return linspace(lambdaZero-(bandWidth/2),lambdaZero+(bandWidth/2),	nb_points)


def doubleArray2numpy(pyOTKArray, size_x, size_y):

	numpyArray = zeros((size_x, size_y), float64)
	for i in range(size_x):
		for j in range(size_y):
			numpyArray[i][j] = pyOTKArray[(size_x*j)+i]

	return numpyArray


def intArray2numpy(pyOTKArray, size_x, size_y):

	numpyArray = zeros((size_x, size_y), int)
	for i in range(size_x):
		for j in range(size_y):
			numpyArray[i][j] = pyOTKArray[(size_x*j)+i]

	return numpyArray


def surf(dataMatrix, meshsize = 0):

	xshape = dataMatrix.shape[0]
	tshape = dataMatrix.shape[1]
	xa = arange(xshape)
	ta = arange(tshape)
	tResampling = int16(linspace(0,tshape-2,meshsize))	
	maxValue = dataMatrix.max()
	normalisation = pow(2,15) / maxValue;

	# Resample the data to get a more lightweight mesh
	if meshsize != 0:
		dataMatrixResampled = zeros([meshsize,meshsize],float)

		# Resample each tResampling temporal profile by a fourier transform method
		for i in arange(meshsize):
			dataMatrixResampled[i] = scipy.signal.resample(dataMatrix[:,tResampling[i]],meshsize)

		tshape = int16(meshsize)
		xshape = int16(meshsize)
		xa = arange(meshsize)
		ta = arange(meshsize)
	else:	
		if (xshape > tshape):
			dataMatrixResampled = zeros([tshape,tshape],float)

			for i in arange(tshape):
				dataMatrixResampled[i] = scipy.signal.resample(dataMatrix[:,i],tshape)

	# Create points
	points = vtkPoints()
	points.SetNumberOfPoints(tshape*tshape)
	for i in arange(1, tshape):
		for j in arange(1, tshape):
			points.InsertPoint(i+tshape*j, ta[i], ta[j], dataMatrixResampled[i,j]); 

	# create quad unstructured grid
	quadgrida = vtkUnstructuredGrid()
	quad = vtkQuad()

	for i in arange(1, tshape):
		for j in arange(1, tshape):
			quad.GetPointIds().SetId(0, (i-1)+tshape*(j-1))
			quad.GetPointIds().SetId(1, (i-1)+tshape*j)
			quad.GetPointIds().SetId(2, i+tshape*j)
			quad.GetPointIds().SetId(3, i+tshape*(j-1))
			quadgrida.InsertNextCell(quad.GetCellType(), quad.GetPointIds())

	quadgrida.SetPoints(points)

	# terminate pipeline with mapper process object
	dataMappera = vtkDataSetMapper()
	dataMappera.SetInput(quadgrida)

	# create an actor
	dataActora = vtkActor()
	dataActora.SetMapper(dataMappera)
	# dataActora.GetProperty().SetRepresentationToWireframe()
	dataActora.GetProperty().SetColor(0.5, 0, 1)

	# create renderer and assign actors to it
	ren1 = vtkRenderer()
	ren1.AddActor(dataActora)
	ren1.SetBackground(1, 1, 1)

	# add renderer to render window
	renWin = vtkRenderWindow()
	renWin.AddRenderer(ren1)
	renWin.SetSize(800, 800)

	# create the interactor
	iren = vtkRenderWindowInteractor()
	iren.SetRenderWindow(renWin)

	# create an interaction style and attach it to the interactor
	style = vtkInteractorStyleTrackballCamera()
	iren.SetInteractorStyle(style)

	# Uncomment for debugging only	
	#return dataMatrixResampled
	
	# start event loop
	iren.Initialize()
	iren.Start()


# Store the results in a FITS files
def store2Fits(filename,dataMatrix):

	storeDataType = ".fits"
	simNumber = 1

	while os.path.exists(filename+str(simNumber)+str(storeDataType)) == True:
		simNumber = simNumber + 1

	champ_fits = pyfits.HDUList()
	hdu = pyfits.PrimaryHDU()
	hdu.data = dataMatrix
	champ_fits.append(hdu)
	champ_fits.writeto(filename+str(simNumber)+".fits")


def store2hdf5(filename,dataMatrix):

	storeDataType = ".h5"
	simNumber = 1

	while os.path.exists(filename+str(simNumber)+str(storeDataType)) == True:
		simNumber = simNumber + 1

	h5file = openFile(filename+str(simNumber)+".h5", mode = "w", title = "Test file")
	root = h5file.createGroup(h5file.root, "genPulse", "genPulse simulation results")
	datasets = h5file.createGroup(root, "simulation", "simulation results")
	h5file.createArray(datasets, 'fieldIntensity', dataMatrix, "Field")
	h5file.close()	


def store2hdf5_b(filename,dataMatrix):

	class Simulation(IsDescription):
		fieldIntensity = FloatCol(shape=(archiveSampling,(2*(archiveTaille+2))))
	
	h5file = openFile(filename+str(simNumber)+".h5", mode = "w", title = "Test file")
	group = h5file.createGroup("/", 'genpulse', 'genPulse results')
	table = h5file.createTable(group, 'readout', Simulation, "Readout example")
	simulation = table.row
	
	simulation['fieldIntensity']  = resultat
	simulation.append()
   
	h5file.close()		


def hdf5load(filename):

	h5file = openFile(filename, 'r')
	datasetObj = h5file.getNode('/genPulse/simulation', 'fieldIntensity')
	datasetArray = datasetObj.read()
	h5file.close()

	return datasetArray


def fitsload(filename):

	hdulist = pyfits.open(filename)
	datasetArray = hdulist[0].data
	return datasetArray	


def movie(dataMatrix, showFrame = 0, scaleAuto = 1):
	
	matrixSize = dataMatrix.shape[1]
	ax = pl.subplot(111)	
	line, = pl.plot(dataMatrix[:,1])

	maxValue = dataMatrix.max()
	ax.set_ylim((0,maxValue))

	if showFrame:
		pl.title("Movie - Frame: " + str(1))

	for i in pl.arange(matrixSize):
	  	line.set_ydata(dataMatrix[:,i])
		if (scaleAuto==1):
			maxValue = dataMatrix[:,i].max()
			ax.set_ylim((0,maxValue))
		if showFrame:
			pl.title("Movie - Frame: " + str(i))
    		pl.draw()


def ossmOutputMP4(movieFilename, dataMatrix1, dataMatrix2, z, kappa, movieFPS = 10, showFrame = 0, scaleAuto = 0):
	'''
	Create a MP4 movie file with the output of the PyOFTK.ossm() function

	Thanks to Josh Lifton
	http://matplotlib.sourceforge.net/examples/animation/movie_demo.html

	'''

	import subprocess
	import os
	import sys
	import matplotlib
	import matplotlib.collections as collections

	matrixSize1 = dataMatrix1.shape[1]
	matrixSize2 = dataMatrix2.shape[1]

	maxValue = (dataMatrix1+dataMatrix2).max()
	kappaMax = kappa.max()

	print "\n\n*** Creation of the avi file: Generation of the frames ***"
	for i in pl.arange(matrixSize1):
		pl.figure(figsize=(19.2,10.8))
		ax = pl.subplot(111)
	  	pl.plot(z,dataMatrix1[:,i]+dataMatrix2[:,i])
		ax.set_ylabel("Intensity [W]")
		ax.set_xlabel("FBG Lenght [mm]")
		ax.set_ylim((0,maxValue))
		ax.set_xlim((z.min(),z.max()))
		collection = collections.BrokenBarHCollection.span_where(z, ymin=0, ymax=maxValue, where=kappa>0, facecolor='blue', alpha=0.05)
		ax.add_collection(collection)

		filename = str('output/%03d' % i) + '.png'
		pl.savefig(filename, dpi=72)
		pl.clf()


	try:
		subprocess.check_call(['mencoder'])
	except subprocess.CalledProcessError:
		print "mencoder command was found"
		pass
	except OSError:
		print "mencoder not found"
		sys.exit("quitting\n")

	command = ('mencoder',
		       'mf://output/*.png',
		       '-mf',
		       'type=png:w=1920:h=1080:fps='+ str(movieFPS),
		       '-ovc',
		       'lavc',
		       '-lavcopts',
		       'vcodec=mpeg4',
		       '-oac',
		       'copy',
		       '-o',
		       movieFilename + '.avi')

	print "\n\n*** Creation of the avi file: Encoding of the video file ***"
	print "\n\nAbout to execute:\n%s\n\n" % ' '.join(command)
	subprocess.check_call(command)

	print "\n\n The movie was written to '" + movieFilename + ".avi'"

	print "\n\nDelete temp files"
	for i in pl.arange(matrixSize1):
		os.remove(os.getcwd()+'/output/'+ str('%03d' % i) +'.png')



def movie_b(dataMatrix, showFrame = 0, nbrLoop = 1):
	
	matrixSize = dataMatrix.shape[1]
	ax = pl.subplot(111)	
	line, = pl.plot(dataMatrix[:,1])

	if showFrame:
		pl.title("Movie - Frame: " + str(1))

	for j in pl.arange(nbrLoop):
		for i in pl.arange(matrixSize):
		  	line.set_ydata(dataMatrix[:,i])
			maxValue = dataMatrix[:,i].max()
			ax.set_ylim((0,maxValue))
			if showFrame:
				pl.title("Movie - Frame: " + str(i))
				pl.draw()

def movie_c(dataMatrix, showFrame):

	matrixSize = dataMatrix.shape[1]
	ax = pl.subplot(111)	
	canvas = ax.figure.canvas
	pl.subplots_adjust(left=0.3, bottom=0.3) 	# check for flipy bugs
	pl.grid()									# to ensure proper background restore
	line, = pl.plot(dataMatrix[:,1])
	tstart = time.time()

	def update_line(*args):
		if update_line.background is None:
		    update_line.background = canvas.copy_from_bbox(ax.bbox)

		# restore the clean slate background
		canvas.restore_region(update_line.background)
		# update the data
		line.set_ydata(dataMatrix[:,1+update_line.cnt])
		# just draw the animated artist
		try:
		    ax.draw_artist(line)
		except AssertionError:
		    return
		# just redraw the axes rectangle
		canvas.blit(ax.bbox)

		if update_line.cnt==matrixSize-2:
		    # print the timing info and quit
		    print 'FPS:' , matrixSize/(time.time()-tstart)
		    sys.exit()

		update_line.cnt += 1
		return True

	update_line.cnt = 0
	update_line.background = None
	gobject.idle_add(update_line)
	pl.show()


def movielog(dataMatrix, showFrame):
	
	matrixSize = dataMatrix.shape[1]
	ax = pl.subplot(111)	
	line, = pl.semilogy(dataMatrix[:,1])

	if showFrame:
		pl.title("Movie - Frame: " + str(1))

	for i in pl.arange(matrixSize):
	  	line.set_ydata(dataMatrix[:,i])
		ax.relim() 
		ax.autoscale_view()
		if showFrame:
			pl.title("Movie - Frame: " + str(i))
    		pl.draw()


def FWHM(time, pulseDataRaw, tol = 1.0):
	'''
	Compute the Full Width at Half Maximun of the pulse in
	the same units as the time vector
	'''
	
	#timeSpl = arange(time[0], time[len(time)-1], splinePts)
	#pulseData_spline = scipy.interpolate.splrep(time,  signal.wiener(pulseData,5,10))
	#pulseData_int = scipy.interpolate.splev(timeSpl, pulseData_spline)
	#pulseData = signal.wiener(pulseDataRaw,500,500)
	dt = time[1]-time[0]
	pulseData = pulseDataRaw
	pulseRange = pulseData.max()-pulseData.min()
	# Tolerance in %
	tolerance = tol*pulseRange/100
	pulseHalf = pulseData.min()+(pulseRange/2)
	halfPoint = ((pulseData>pulseHalf-tolerance) & (pulseData<pulseHalf+tolerance)).nonzero()
	return time[halfPoint[0].max()] - time[halfPoint[0].min()]



def effWidth(time, pulseDataRaw, tol = 1.0):
	'''
	Compute the effective width of the pulse in
	the same units as the time vector
	'''
	cstHonte = 8.24/11.189
	dt = time[1]-time[0]
	pulseData = pulseDataRaw
	return cstHonte*(pulseData.sum() / pulseData.max())*dt


def skewness(time, pulseInt):
	'''
	Compute the skew parameter of the pulse
	c.f. bale2010impact
	'''
	A = sqrt(pulseInt/pulseInt.max())
	tau = time / effWidth(time, pulseInt)
	dtau=tau[1]-tau[0]
	arg=pow(tau,3.0)*pow(A,2.0)

	return arg.sum()*dtau


def skewness3(t, pulseInt):
	'''
	Compute the skew parameter of the pulse
	c.f. bale2010impact
	'''
	maxPos = where(pulseInt==pulseInt.max())
	t2 = t - t[maxPos]
	A = sqrt(pulseInt/pulseInt.max())
	tau = t2 / effWidth(t2, pulseInt)
	dtau=tau[1]-tau[0]
	arg=pow(tau,3.0)*pow(A,2.0)

	return arg.sum()*dtau


def normalize(t, pulseInt):
	maxValue = pulseInt.max()
	maxPos = where(pulseInt==maxValue)
	t2 = t - t[maxPos]
	tau = t2 / effWidth(t2, pulseInt)
	dtau=tau[1]-tau[0]
	return [tau, pulseInt/maxValue]


def normalizeInt(t, pulseInt):
	maxValue = pulseInt.max()
	maxPos = where(pulseInt==maxValue)

	return pulseInt/maxValue


def recenter(t, pulseInt):
	maxPos = where(pulseInt==pulseInt.max())[0][0]
	t2 = t - t[maxPos]
	return t2


def moment(t, pulse, k):
	dt = t[1]-t[0]
	return ( (pow(t,k)*pulse).sum().real*dt ) / ( pulse.sum().real*dt )


def skewness2(t, pulseInt):
	'''
	Define a skew parameter in stat way
	third moment / second moment ^ 3/2
	'''
	return moment(t, pulseInt, 3.0)/(pow(moment(t, pulseInt, 2.0),3.0/2))


def parabolicMisFit(t, pulseInt, returnFit = 0):
	'''
	Compute the MisFit of the pulseInt profil with a parabolic pulse profil
		* t: time vector
		* pulseInt: pulse intensity profile
		* returnFit: Return the parabolic fit if = 1 (default = 0)
	'''
	A = pulseInt/pulseInt.max()
	tau = t / effWidth(t, pulseInt)
	dtau=tau[1]-tau[0]
	fit = pulseIntFit(tau, A, shape='parabolic')
	m_squared = (pow((A - fit),2)).sum()*dtau/(pow(A,2)*dtau).sum()
	if returnFit:
		return [m_squared, fit]
	else:	
		return m_squared


def pulseIntMisFit(t, pulseIntA, pulseIntB):
	'''
	Compute the m_squared misfit between two pulses intensity profile
	'''
	dt = t[1]-t[0]
	m_squared = (pow((pulseIntA - pulseIntB),2)).sum()*dt/(pow(pulseIntA,2)*dt).sum()
	return m_squared


def gaussianMisFit(t, pulseInt, returnFit = 0):
	'''
	Compute the MisFit of the pulseInt profil with a gaussian pulse profil
		* t: time vector
		* pulseInt: pulse intensity profile
		* returnFit: Return the parabolic fit if = 1 (default = 0)
	'''
	A = pulseInt/pulseInt.max()
	tau = t / effWidth(t, pulseInt)
	dtau=tau[1]-tau[0]
	fit = pulseIntFit(tau, A, shape='gaussian')
	m_squared = (pow((A - fit),2)).sum()*dtau/(pow(A,2)*dtau).sum()
	if returnFit:
		return [m_squared, fit]
	else:	
		return m_squared


def sechMisFit(t, pulseInt, returnFit = 0):
	'''
	Compute the MisFit of the pulseInt profil with a gaussian pulse profil
		* t: time vector
		* pulseInt: pulse intensity profile
		* returnFit: Return the parabolic fit if = 1 (default = 0)
	'''
	A = pulseInt/pulseInt.max()
	tau = t / effWidth(t, pulseInt)
	dtau=tau[1]-tau[0]
	fit = pulseIntFit(tau, A, shape='sech')
	m_squared = (pow((A - fit),2)).sum()*dtau/(pow(A,2)*dtau).sum()
	if returnFit:
		return [m_squared, fit]
	else:	
		return m_squared


def parabolicEffWidth(t, pulseDataRaw):
	'''
	Compute the effective width of a parabolic pulse in
	the same units as the time vector

	cf. Fermann et al.
	'''
	
	dt = t[1]-t[0]
	pulseData = pulseDataRaw
	fit = pulseIntFit(t, pow(abs(pulseDataRaw),2), shape='parabolic')

	return (where(fit>0.0)[0].max() - where(fit>0.0)[0].min())*dt


def ptsFWHM(pulseDataRaw, tol = 1.0):
	'''
	Compute the Full Width at Half Maximun of the pulse in
	number of points
	'''
	
	pulseData = pulseDataRaw
	pulseRange = pulseData.max()-pulseData.min()
	# Tolerance in %
	tolerance = tol*pulseRange/100
	pulseHalf = pulseData.min()+(pulseRange/2)
	halfPoint = ((pulseData>pulseHalf-tolerance) & (pulseData<pulseHalf+tolerance)).nonzero()
	return halfPoint[0].max()-halfPoint[0].min()


def rmsWidth(time, pulseDataRaw):
	'''
	Compute the rms Width of the pulse in the same units
	of the time vector
	'''

	dt = time[1]-time[0]
	moment2 = ( (pow(time,2)*pulseDataRaw).sum().real*dt ) / ( pulseDataRaw.sum().real*dt )
	moment1 = ( (time*pulseDataRaw).sum().real*dt ) / ( pulseDataRaw.sum().real*dt )
	return sqrt(moment2 - pow(moment1,2)).real


def intAC(eFieldSVEA):
	'''
	Compuite the optical autocorrelation trace
	'''
	AC = signal.correlate(eFieldSVEA, eFieldSVEA, mode='same')	
	
	return AC


def plotChirp(t, eFieldSVEA, filtering = [2000,500]):
	'''
	Plot the frequency chirp of a pulse
		* t: time vector
		* SVEA Amplitude Field of the pulse
	'''
	
	nt = len(eFieldSVEA)
	nuInstPulse = signal.wiener(nuInst(eFieldSVEA),filtering[0], filtering[1])
	width = ptsFWHM(pow(abs(eFieldSVEA),2))

	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax2 = ax1.twinx()

	y1, y2 = ax1.get_ylim()
	width = FWHM(t, pow(abs(eFieldSVEA),2))
	ax2.figure.canvas.draw()

	ax1.plot(pow(abs(eFieldSVEA),2))
	ax2.plot(nuInstPulse, color='red')
	#ax2.set_ylim([nuInstPulse[(nt/2)-width], nuInstPulse[(nt/2)+width]])


def laserPlot(filenameLst):
	'''
	Plot the temporal and spectral profile of a
	mode-locked fiber laser simulation
	'''

	nbrPlots = len(filenameLst)
	for i in arange(len(filenameLst)):
		results = load(filenameLst[i])
		t = results['t']
		nt = results['nt']
		T = results['T']
		archivePass = results['archivePass']
		nu_inst_out3 = results['nu_inst_out3']
		nu_inst_out4 = results['nu_inst_out4']
		spectre_out = results['spectre_out']
		wavelength = results['wavelength']


		# Graph
		plt.figure(figsize=(12,9))

		ax3 = plt.subplot(221)
		plt.plot(t, pow(abs(archivePass[0]),2), color="black")
		plt.ylabel("$|u(z,T)|^2$ [W]")
		plt.xlabel("$T/T_0$")
		plt.xlim([-T/2,T/2])
		plt.grid(True)
		ax4 = plt.twinx()
		plt.plot(t[0:nt-1], nu_inst_out3)
		plt.ylabel("Chirp")
		ax4.yaxis.tick_right()
		plt.ylim([-1.5,1.5])

		ax5 = plt.subplot(223)
		plt.semilogy(t, pow(abs(archivePass[0]),2), color="black")
		plt.ylabel("$|u(z,T)|^2$ [dBm]")
		plt.xlabel("$T/T_0$")
		plt.xlim([-T/2,T/2])
		plt.grid(True)
		ax4 = plt.twinx()
		plt.plot(t[0:nt-1], nu_inst_out3)
		plt.ylabel("Chirp")
		ax4.yaxis.tick_right()
		plt.ylim([-1.5,1.5])

		ax7 = plt.subplot(222)
	 	plt.plot(wavelength, spectre_out, color="black")
		plt.xlabel("Wavelength [nm]")
		plt.grid(True)

		ax8 = plt.subplot(224)
	 	plt.semilogy(wavelength, spectre_out, color="black")
		plt.xlabel("$T/T_0$")
		plt.xlabel("Wavelength [nm]")
		plt.grid(True)

	plt.show()


def pulseIntFit(t, pulseInt, shape='gaussian'):
	'''
	Fit the pulse with a specific pulse shape
		* t: time vector
		* pulseInt: Pulse intensity
		* shape: Pulse shape ('gaussian', 'sech', 'parabolic')
	'''

	# Distance to the target function
	errfunc_g = lambda p, t, y: pulse.gaussianPulseCmpt(t, p) - y
	errfunc_s = lambda p, t, y: pulse.sechPulseCmpt(t, p) - y
	errfunc_p = lambda p, t, y: pulse.parabolicPulseFitCmpt(t, p) - y

	# Initial guess for the parameters
	p0_g = [5.0, 0.0, 1.0, 0.0]
	p0_s = [5.0, 0.0, 1.0, 0.0]
	p0_p = [5.0, 0.0, pulseInt.max()]

	# Fit with rge least square method
	[p, success]  = {
	  'gaussian': lambda: leastsq(errfunc_g, p0_g[:], args=(t, pulseInt)),
	  'sech': lambda: leastsq(errfunc_s, p0_s[:], args=(t, pulseInt)),
	  'parabolic': lambda: leastsq(errfunc_p, p0_p[:], args=(t, pulseInt)),
	}[shape]()

	fit  = {
	  'gaussian': lambda: pulse.gaussianPulseCmpt(t, p),
	  'sech': lambda: pulse.sechPulseCmpt(t, p),
	  'parabolic': lambda: pulse.parabolicPulseFitCmpt(t, p),
	}[shape]()

	return fit


def fitLaserResults(filename):

	results = load(filename)
	t = results['t']
	nt = results['nt']
	T = results['T']
	archivePass = results['archivePass']
	nu_inst_out3 = results['nu_inst_out3']
	nu_inst_out4 = results['nu_inst_out4']
	spectre_out = results['spectre_out']
	u_int = pow(abs(archivePass[0]),2)

	def sechPulse(t, p):
		'''
		This function computes a hyperbolic secant pulse with the
		specified parameters:
		'''

		FWHM = p[0]
		t0 = p[1]
		P0 = p[2]
		C = p[3]
		T_zero = FWHM/(2*arccosh(sqrt(2)));
		return pow( abs(sqrt(P0)*1/cosh((t-t0)/T_zero)*exp(-1j*C*pow((t-t0),2)/(2*pow(T_zero,2)))), 2)


	def gaussianPulse(t, p):
		"""
		Geneate a gaussian/supergaussiance envelope pulse

			* field_amp: 	output gaussian pulse envellope (amplitude).
			* t:     		vector of times at which to compute u
			* t0:    			center of pulse (default = 0)
			* FWHM:   		full-width at half-intensity of pulse (default = 1)
			* P0:    		peak intensity of the pulse @ t=t0 (default = 1)
			* m:     		Gaussian order (default = 1)
			* C:     		chirp parameter (default = 0)
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


	def parabolicPulse(t, t0, P0):
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


	def parabolicPulseFit(t, p):
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
	
		parabolicFit = parabolicPulse(t, t0, P0)

		while abs( parabolicFit[FWHM_Right] - (P0/2) ) > 1:
			if (parabolicFit[FWHM_Right] < (P0/2) ):
				parabolicFit = parabolicPulse(t/scalingFactor, t0, P0)
			else:
				parabolicFit = parabolicPulse(t*scalingFactor, t0, P0)
			scalingFactor = scalingFactor + 0.01

		return parabolicFit


	# Distance to the target function
	errfunc = lambda p, t, y: gaussianPulse(t, p) - y
	errfuncb = lambda p, t, y: sechPulse(t, p) - y
	errfuncc = lambda p, t, y: parabolicPulseFit(t, p) - y

	# Initial guess for the parameters
	p0 = [5.0, 0.0, 1.0, 0.0]
	p0b = [5.0, 0.0, 1.0, 0.0]
	p0c = [5.0, 0.0, 1.0]

	p1, success = leastsq(errfunc, p0[:], args=(t, u_int))
	p1b, success = leastsq(errfuncb, p0b[:], args=(t, u_int))
	p1c, success = leastsq(errfuncc, p0c[:], args=(t, u_int))

	print "\n*** Gaussian fit parameters ***"
	print "FWHM: " + str(p1[0])
	print "t0: " + str(p1[1])
	print "P0: " + str(p1[2])
	print "C: " + str(p1[3])

	print "\n***** Sech fit parameters *****"
	print "FWHM: " + str(p1b[0])
	print "t0: " + str(p1b[1])
	print "P0: " + str(p1b[2])
	print "C: " + str(p1b[3])

	fit = gaussianPulse(t, p1)
	fitb = sechPulse(t, p1b)
	fitc = parabolicPulseFit(t, p1c)

	# Error function of the optimal fit
	#err = errfunc(p1, t, u_int)
	#err_b = errfuncb(p1b, t, u_int)


	plt.subplot(1,2,1)
	plt.plot(t, u_int, t, fit, t, fitb)
	plt.grid(True)
	plt.xlabel('Temps [ps]')
	plt.ylabel('Intensite [W]')
	plt.legend(("Simulation","Curve fit (gaussienne)","Curve fit (sech)"))

	plt.subplot(1,2,2)
	plt.semilogy(t, u_int, t, fit, t, fitb)
	plt.grid(True)
	plt.xlabel('Temps [ps]')
	plt.ylabel('Intensite [W]')
	plt.legend(("Simulation","Fit (gaussienne)","Fit (sech)"))
	plt.show()


def fitPeakLaserResults(filename):

	results = load(filename)
	t = results['t']
	nt = results['nt']
	T = results['T']
	archivePass = results['archivePass']
	nu_inst_out3 = results['nu_inst_out3']
	nu_inst_out4 = results['nu_inst_out4']
	spectre_out = results['spectre_out']
	u_int = pow(abs(archivePass[0]),2)

	def sechPulse(t, p):
		'''
		This function computes a hyperbolic secant pulse with the
		specified parameters:
		'''

		FWHM = p[0]
		t0 = p[1]
		P0 = p[2]
		C = p[3]
		T_zero = FWHM/(2*arccosh(sqrt(2)));
		return pow( abs(sqrt(P0)*1/cosh((t-t0)/T_zero)*exp(-1j*C*pow((t-t0),2)/(2*pow(T_zero,2)))), 2)


	def gaussianPulse(t, p):
		"""
		Geneate a gaussian/supergaussiance envelope pulse

			* field_amp: 	output gaussian pulse envellope (amplitude).
			* t:     		vector of times at which to compute u
			* t0:    			center of pulse (default = 0)
			* FWHM:   		full-width at half-intensity of pulse (default = 1)
			* P0:    		peak intensity of the pulse @ t=t0 (default = 1)
			* m:     		Gaussian order (default = 1)
			* C:     		chirp parameter (default = 0)
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


	def parabolicPulse(t, t0, P0):
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


	def parabolicPulseFit(t, p):
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
	
		parabolicFit = parabolicPulse(t, t0, P0)

		while abs( parabolicFit[FWHM_Right] - (P0/2) ) > 1:
			if (parabolicFit[FWHM_Right] < (P0/2) ):
				parabolicFit = parabolicPulse(t/scalingFactor, t0, P0)
			else:
				parabolicFit = parabolicPulse(t*scalingFactor, t0, P0)
			scalingFactor = scalingFactor + 0.01

		return parabolicFit


	### Preliminary gaussian fit on the whole pulse ###
	errfuncPre = lambda p, t, y: gaussianPulse(t, p) - y
	p0Pre = [5.0, 0.0, 1.0, 0.0]
	p1Pre, success = leastsq(errfuncPre, p0Pre[:], args=(t, u_int))

	FWHM = p1Pre[0]
	t0 = p1Pre[1]
	peakStart = t0-FWHM/2
	peakStop = t0+FWHM/2
	peakValue = nonzero((t > peakStart) & (t < peakStop))
	tpeak = t[peakValue]
	u_int_peak = u_int[peakValue]
	print tpeak

	### Fit on the peak value only ###

	# Distance to the target function
	errfunc = lambda p, t, y: gaussianPulse(t, p) - y
	errfuncb = lambda p, t, y: sechPulse(t, p) - y
	errfuncc = lambda p, t, y: parabolicPulseFit(t, p) - y

	# Initial guess for the parameters
	p0 = [5.0, 0.0, 1.0, 0.0]
	p0b = [5.0, 0.0, 1.0, 0.0]
	p0c = [5.0, 0.0, 1.0]

	p1, success = leastsq(errfunc, p0[:], args=(tpeak, u_int_peak))
	p1b, successb = leastsq(errfuncb, p0b[:], args=(tpeak, u_int_peak))
	p1c, successc = leastsq(errfuncc, p0c[:], args=(tpeak, u_int_peak))

	fit = gaussianPulse(tpeak, p1)
	fitb = sechPulse(tpeak, p1b)
	fitc = parabolicPulseFit(tpeak, p1c)

	plt.subplot(1,2,1)
	plt.plot(tpeak, u_int_peak, tpeak, fit, tpeak, fitb, tpeak, fitc)
	plt.grid(True)
	plt.xlabel('Temps [ps]')
	plt.ylabel('Intensite [W]')
	plt.legend(("Simulation","Curve fit (gaussienne)","Curve fit (sech)", "Curve fit (parabolic)"))

	plt.subplot(1,2,2)
	plt.semilogy(tpeak, u_int_peak, tpeak, fit, tpeak, fitb, tpeak, fitc)
	plt.grid(True)
	plt.xlabel('Temps [ps]')
	plt.ylabel('Intensite [W]')
	plt.legend(("Simulation","Curve fit (gaussienne)","Curve fit (sech)", "Curve fit (parabolic)"))
	plt.show()




