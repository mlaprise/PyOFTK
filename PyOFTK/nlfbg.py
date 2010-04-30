'''

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


'''

from numpy import *
from scipy import *
from utilities import *

def shift(u, shift = 1):
	size = len(u)
	shift = -shift
	if shift >= 0:
		return r_[zeros(abs(shift),complex),u[0:size-shift]]
	else:
		return r_[u[abs(shift):size],zeros(abs(shift),complex)]

def shiftb(u, shift = 1):
	size = len(u)
	shift = -shift
	if shift >= 0:
		return numpy.r_[numpy.zeros(abs(shift),complex),u[0:size-shift]]
	else:
		return numpy.r_[u[abs(shift):size],numpy.zeros(abs(shift),complex)]


def shiftfourier(u, h, dz, nz):
	k = wspace(dz*nz,nz)
	size = len(u)
	ufft = fftpack.fft(u)
	return fftpack.ifft(ufft*exp(1.0j*k*h))


def ossmtest(up, length, nt, alpha, beta1, kappa, detuning, gamma, stepArchive = 1, dz_over_dt = 1.0, posDectector = 0):

	'''	
	Implementation de l'algorithme Split-Step pour resoudre les 
	equations nonlin. couplees d'une reseau de bragg.
	(version nonsymmetrized)

	ref. Toroker et al. 'Optimized split-step method for modeling
	nonlinear pulse propagation in nonlinear fiber Bragg gratings'

	up: 		Profil spatial de l'impulsion
	length:		Longueur du fbg
	nt:			Nombre de steps temporel
	alpha:		Parametre de gain
	beta1:		Constante de propagation (1/vitesse de groupe)
	kappa:		Parametre de couplage du fbg [1/m]
	detuning:	Detuning par rapport a la longueur d'onde de bragg [1/m]
	'''

	# Vitesse de groupe dans le fibre sans fbg [m/ps]
	Vg = (1.0/beta1)/(1e12)
	nz = len(up)
	dz = length / nz
	k = wspace(dz*nz,nz)

	# Cas simple ou dt est fixee a dz par la vitesse de groupe [ps]
	dt = dz / Vg
	h = dz

	# On suppose que u_moins = 0 a z = L
	um = zeros(nz, complex)
	upArchiveInt = zeros([nt/stepArchive,nz], double)
	umArchiveInt = zeros([nt/stepArchive,nz], double)
	timeDetector = zeros(nt, complex)

	# Construction de l'operateur lineaire
	Dmg = exp(h*(-alpha/2.0))
	Dm1 = (cos(detuning*h)+1.0j*sin(detuning*h))*cos(kappa*h)
	Dm2 = (cos(detuning*h)+1.0j*sin(detuning*h))*1.0j*sin(kappa*h)
	Dp1 = (cos(detuning*h)+1.0j*sin(detuning*h))*1.0j*sin(kappa*h)
	Dp2 = (cos(detuning*h)+1.0j*sin(detuning*h))*cos(kappa*h)
	
	archiveFlag = stepArchive
	ia = 0

	for it in arange(nt):
		# Application de l'operateur nonlineaire
		Nm = 1.0j*gamma*(pow(abs(shift(um,1)),2)+2*pow(abs(shift(up,1)),2))
		Np = 1.0j*gamma*(pow(abs(shift(up,-1)),2)+2*pow(abs(shift(um,-1)),2))
		um_nl = exp(h*Nm)*shift(um,1)
		up_nl = exp(h*Np)*shift(up,-1)

		# Application de l'operateur lineaire
		um_next = (Dmg * Dm1 * um_nl) + (Dmg * Dm2 * up_nl)
		up_next = (Dmg * Dp1 * um_nl) + (Dmg * Dp2 * up_nl)

		# Simulate the effect of the float truncation
		up.real = array(up_next.real, float32)
		up.imag = array(up_next.imag, float32)
		um.real = array(um_next.real, float32)
		um.imag = array(um_next.imag, float32)

		# Store the temporel profile with a "Scifi" amplitude detector
		timeDetector[it] = up[posDectector]
		
		# Archive le signal avec une periode stepArchive
		if archiveFlag == stepArchive:
			upArchiveInt[ia] = pow(abs(up),2)
			umArchiveInt[ia] = pow(abs(um),2)
			ia += 1
			archiveFlag = 1
		else:
			archiveFlag += 1

	return [up, um, upArchiveInt, umArchiveInt, timeDetector]


def ossm(up, length, nt, alpha, beta1, kappa, detuning, gamma, stepArchive = 1, dz_over_dt = 1.0, posDectector = 0):

	'''	
	Implementation of the optimized split-step method for solving nonlinear 
	coupled-mode equations that model wave propagation in nonlinear fiber Bragg
	gratings. (version nonsymmetrized)

	-- CPU Version -- 

	ref. Toroker et al. 'Optimized split-step method for modeling
	nonlinear pulse propagation in nonlinear fiber Bragg gratings'

	up: 		Profil spatial de l'impulsion
	length:		Longueur du fbg
	nt:			Nombre de steps temporel
	alpha:		Parametre de gain
	beta1:		Constante de propagation (1/vitesse de groupe)
	kappa:		Parametre de couplage du fbg [1/m]
	detuning:	Detuning par rapport a la longueur d'onde de bragg [1/m]
	'''

	# Vitesse de groupe dans le fibre sans fbg [m/ps]
	Vg = (1.0/beta1)/(1e12)
	nz = len(up)
	dz = length / nz
	k = wspace(dz*nz,nz)

	# Cas simple ou dt est fixee a dz par la vitesse de groupe [ps]
	dt = dz / Vg
	h = dz

	# On suppose que u_moins = 0 a z = L
	um = zeros(nz, complex)
	upArchiveInt = zeros([nt/stepArchive,nz], double)
	umArchiveInt = zeros([nt/stepArchive,nz], double)
	timeDetector = zeros(nt, complex)

	# Construction de l'operateur lineaire
	Dmg = exp(h*(-alpha/2.0))
	Dm1 = (cos(detuning*h)+1.0j*sin(detuning*h))*cos(kappa*h)
	Dm2 = (cos(detuning*h)+1.0j*sin(detuning*h))*1.0j*sin(kappa*h)
	Dp1 = (cos(detuning*h)+1.0j*sin(detuning*h))*1.0j*sin(kappa*h)
	Dp2 = (cos(detuning*h)+1.0j*sin(detuning*h))*cos(kappa*h)
	
	archiveFlag = stepArchive
	ia = 0

	for it in arange(nt):
		# Application de l'operateur nonlineaire
		Nm = 1.0j*gamma*(pow(abs(shift(um,1)),2)+2*pow(abs(shift(up,1)),2))
		Np = 1.0j*gamma*(pow(abs(shift(up,-1)),2)+2*pow(abs(shift(um,-1)),2))
		um_nl = exp(h*Nm)*shift(um,1)
		up_nl = exp(h*Np)*shift(up,-1)

		# Application de l'operateur lineaire
		um_next = (Dmg * Dm1 * um_nl) + (Dmg * Dm2 * up_nl)
		up_next = (Dmg * Dp1 * um_nl) + (Dmg * Dp2 * up_nl)

		up = up_next
		um = um_next

		# Store the temporel profile with a "Scifi" amplitude detector
		timeDetector[it] = up[posDectector]
		
		# Archive le signal avec une periode stepArchive
		if archiveFlag == stepArchive:
			upArchiveInt[ia] = pow(abs(up),2)
			umArchiveInt[ia] = pow(abs(um),2)
			ia += 1
			archiveFlag = 1
		else:
			archiveFlag += 1

	return [up, um, upArchiveInt, umArchiveInt, timeDetector]
	

def partialStep(up, um, i, j, h, gamma, Dmg, Dm1, Dm2, Dp1, Dp2):

		# Application de l'operateur nonlineaire
		Nm = 1.0j*gamma*(pow(abs(shiftb(um[i:j],1)),2)+2*pow(abs(shiftb(up[i:j],1)),2))
		Np = 1.0j*gamma*(pow(abs(shiftb(up[i:j],-1)),2)+2*pow(abs(shiftb(um[i:j],-1)),2))
		um_nl = numpy.exp(h*Nm)*shiftb(um[i:j],1)
		up_nl = numpy.exp(h*Np)*shiftb(up[i:j],-1)

		# Application de l'operateur lineaire
		#print str(len((Dmg[i:j] * Dm1[i:j] * um_nl) + (Dmg * Dm2[i:j] * up_nl)))
		um_next = (Dmg[i:j] * Dm1[i:j] * um_nl) + (Dmg[i:j] * Dm2[i:j] * up_nl)
		up_next = (Dmg[i:j] * Dp1[i:j] * um_nl) + (Dmg[i:j] * Dp2[i:j] * up_nl)

		return [up_next, um_next]


def ossmsmp(up, length, nt, alpha, beta1, kappa, detuning, gamma, stepArchive = 1, dz_over_dt = 1.0, posDectector = 0):

	import pp
	
	'''	
	Implementation de l'algorithme Split-Step pour resoudre les 
	equations nonlin. couplees d'une reseau de bragg.
	(version nonsymmetrized)

	ref. Toroker et al. 'Optimized split-step method for modeling
	nonlinear pulse propagation in nonlinear fiber Bragg gratings'

	 -- Version SMP -- 

	up: 		Profil spatial de l'impulsion
	length:		Longueur du fbg
	nt:			Nombre de steps temporel
	alpha:		Parametre de gain
	beta1:		Constante de propagation (1/vitesse de groupe)
	kappa:		Parametre de couplage du fbg [1/m]
	detuning:	Detuning par rapport a la longueur d'onde de bragg [1/m]

	'''

	# Initialisation of the smp stuff (parallel python)
	ppservers = ()
	ncpus = 4
	job_server = pp.Server(ncpus, ppservers=ppservers)
	print "Starting ossm simulation with", job_server.get_ncpus(), "CPUs"

	# Vitesse de groupe dans le fibre sans fbg [m/ps]
	Vg = (1.0/beta1)/(1e12)
	nz = len(up)
	dz = length / nz
	k = wspace(dz*nz,nz)

	# Set the granularity of the pp execution
	parts = ncpus
	step = nz/parts
	jobs = []

	# Cas simple ou dt est fixee a dz par la vitesse de groupe [ps]
	dt = dz / Vg
	h = dz

	# On suppose que u_moins = 0 a z = L
	um = zeros(nz, complex)
	up_next = zeros(nz, complex)
	um_next = zeros(nz, complex)
	upArchiveInt = zeros([nt/stepArchive,nz], double)
	umArchiveInt = zeros([nt/stepArchive,nz], double)
	timeDetector = zeros(nt, complex)

	# Construction de l'operateur lineaire
	Dmg = exp(h*(-alpha/2.0))
	Dm1 = (cos(detuning*h)+1.0j*sin(detuning*h))*cos(kappa*h)
	Dm2 = (cos(detuning*h)+1.0j*sin(detuning*h))*1.0j*sin(kappa*h)
	Dp1 = (cos(detuning*h)+1.0j*sin(detuning*h))*1.0j*sin(kappa*h)
	Dp2 = (cos(detuning*h)+1.0j*sin(detuning*h))*cos(kappa*h)
	
	archiveFlag = stepArchive
	ia = 0

	for it in arange(nt):

		for index in arange(parts):
			starti = index*step
			endi = min((index+1)*step, nz)
			jobs.append(job_server.submit(partialStep, (up, um, starti, endi, h, gamma, Dmg, Dm1, Dm2, Dp1, Dp2,), (shiftb,), ("PyOFTK","numpy")))

		for job in jobs:
			result = job()
			if result:
				break

		# Store the temporel profile with a "Scifi" amplitude detector
		timeDetector[it] = up[posDectector]
		
		# Archive le signal avec une periode stepArchive
		if archiveFlag == stepArchive:
			upArchiveInt[ia] = pow(abs(up),2)
			umArchiveInt[ia] = pow(abs(um),2)
			ia += 1
			archiveFlag = 1
		else:
			archiveFlag += 1

	return [up, um, upArchiveInt, umArchiveInt, timeDetector]


def ossmgpu(up, length, nt, alpha, beta1, kappa, detuning, gamma, stepArchive = 1, dz_over_dt = 1.0, posDectector = 0):
	'''	
	Implementation of the optimized split-step method for solving nonlinear 
	coupled-mode equations that model wave propagation in nonlinear fiber Bragg
	gratings. (version nonsymmetrized)

	ref. Toroker et al. 'Optimized split-step method for modeling
	nonlinear pulse propagation in nonlinear fiber Bragg gratings'

	 -- CUDA Version -- 

	up: 		Profil spatial de l'impulsion
	length:		Longueur du fbg
	nt:			Nombre de steps temporel
	alpha:		Parametre de gain
	beta1:		Constante de propagation (1/vitesse de groupe)
	kappa:		Parametre de couplage du fbg [1/m]
	detuning:	Detuning par rapport a la longueur d'onde de bragg [1/m]
	'''


	import pycuda.driver as cuda
	import pycuda.compiler as cudaComp
	import pycuda.autoinit


	# Vitesse de groupe dans le fibre sans fbg [m/ps]
	Vg = (1.0/beta1)/(1e12)
	nz = len(up)
	dz = length / nz
	k = wspace(dz*nz,nz)

	# Cas simple ou dt est fixee a dz par la vitesse de groupe [ps]
	dt = dz / Vg
	h = dz

	# On suppose que u_moins = 0 a z = L
	um = zeros(nz, complex)
	upArchiveInt = zeros([nt/stepArchive,nz], double)
	umArchiveInt = zeros([nt/stepArchive,nz], double)
	upArchiveInt32 = zeros([nt/stepArchive,nz], float32)
	umArchiveInt32 = zeros([nt/stepArchive,nz], float32)
	up_im_ArchiveInt32 = zeros([nt/stepArchive,nz], float32)
	um_im_ArchiveInt32 = zeros([nt/stepArchive,nz], float32)
	timeDetector = zeros(nt, complex)
	timeDetector32 = zeros(nt, float32)
	timeDetector32_im = zeros(nt, float32)

	# Matrice qui va contenir le stuff a envoyer au gpu
	operatorArray = zeros([6,nz], float32)
	operatorArray_im = zeros([6,nz], float32)

	# Construction de l'operateur lineaire
	Dmg = exp(h*(-alpha/2.0))
	Dm1 = (cos(detuning*h)+1.0j*sin(detuning*h))*cos(kappa*h)
	Dm2 = (cos(detuning*h)+1.0j*sin(detuning*h))*1.0j*sin(kappa*h)
	Dp1 = (cos(detuning*h)+1.0j*sin(detuning*h))*1.0j*sin(kappa*h)
	Dp2 = (cos(detuning*h)+1.0j*sin(detuning*h))*cos(kappa*h)

	# Transfert des operateurs et parametres
	operatorArray[0,0] = float(nz)
	operatorArray[0,1] = nt
	operatorArray[0,2] = gamma
	operatorArray[0,3] = h
	operatorArray[0,4] = posDectector

	operatorArray[1] = Dm1.real.astype(float32)
	operatorArray_im[1] = Dm1.imag.astype(float32)
	operatorArray[2] = Dm2.real.astype(float32)
	operatorArray_im[2] = Dm2.imag.astype(float32)
	operatorArray[3] = Dp1.real.astype(float32)
	operatorArray_im[3] = Dp1.imag.astype(float32)
	operatorArray[4] = Dp2.real.astype(float32)
	operatorArray_im[4] = Dp2.imag.astype(float32)
	operatorArray[5] = exp(h*(-alpha/2.0))

	paramatersInt = zeros(1, int)
	
	archiveFlag = stepArchive
	ia = 0

	##########################################
	### Initialisation of the PyCUDA Stuff ###
	##########################################
	
	BlockSize = 3
	operatorArray[0,5] = BlockSize
	dimBlock = (BlockSize,1,1)
	dimGrid	= (nz/BlockSize,1)

	# Convert to float32
	up_float32 = up.real.astype(float32)
	up_float32_im = up.imag.astype(float32)
	um_float32 = um.real.astype(float32)
	um_float32_im = um.imag.astype(float32)

	# Allocation and copy to the gpu memory
	opArray_gpu = cuda.mem_alloc(operatorArray.size*operatorArray.dtype.itemsize)
	cuda.memcpy_htod(opArray_gpu, operatorArray)
	opArray_im_gpu = cuda.mem_alloc(operatorArray_im.size*operatorArray_im.dtype.itemsize)
	cuda.memcpy_htod(opArray_im_gpu, operatorArray_im)

	timeDetector_gpu = cuda.mem_alloc(timeDetector32.size*timeDetector32.dtype.itemsize)
	cuda.memcpy_htod(timeDetector_gpu, timeDetector32)
	timeDetector_im_gpu = cuda.mem_alloc(timeDetector32_im.size*timeDetector32_im.dtype.itemsize)
	cuda.memcpy_htod(timeDetector_im_gpu, timeDetector32_im)

	up_gpu = cuda.mem_alloc(up_float32.size*up_float32.dtype.itemsize)
	cuda.memcpy_htod(up_gpu, up_float32)
	up_im_gpu = cuda.mem_alloc(up_float32_im.size*up_float32_im.dtype.itemsize)
	cuda.memcpy_htod(up_im_gpu, up_float32_im)

	um_gpu = cuda.mem_alloc(um_float32.size*um_float32.dtype.itemsize)
	cuda.memcpy_htod(um_gpu, um_float32)
	um_im_gpu = cuda.mem_alloc(um_float32_im.size*um_float32_im.dtype.itemsize)
	cuda.memcpy_htod(um_im_gpu, um_float32_im)


	######### Optimized SplitStep Method Kernel	#########
	#####################################################

	mod = cudaComp.SourceModule("""

	  __global__ void splitstep(float *up_re, float *up_im, float *um_re, float *um_im, float *opArray_re, float *opArray_im, float *timeDetector, float *timeDetector_im)
	  {

		int idxs;
		int is;
		const uint blockSize = 20;
		int idx = blockIdx.x;

		// Get the simulation parameters store in the unused part of opArray_re
		int nz = (int)opArray_re[0];
		float gamma = opArray_re[2];
		float h = opArray_re[3];

		float Nm_im;
		float Np_im;
		float um_nl_im;
		float um_nl_re;
		float up_nl_im;
		float up_nl_re;

		float up_next_im;
		float up_next_re;
		float um_next_im;
		float um_next_re;

		// Shared memory
		// up_re = u[0], up_im = u[1], um_re = u[2], um_im = u[3]
		__shared__ float u[4][3];
		__shared__ float opArray_re_s[6][3];
		__shared__ float opArray_im_s[6][3];

		// Device memory to shared memory
		for(is = 0; is < 3; is++)
		{
			idxs = (idx*blockSize) + is;

			u[0][is] =  up_re[idxs];
			u[1][is] =  up_im[idxs];
			u[2][is] =  um_re[idxs];
			u[3][is] =  um_im[idxs];

			opArray_re_s[0][is] = opArray_re[idxs];
			opArray_re_s[1][is] = opArray_re[idxs+1*nz];
			opArray_re_s[2][is] = opArray_re[idxs+2*nz];
			opArray_re_s[3][is] = opArray_re[idxs+3*nz];
			opArray_re_s[4][is] = opArray_re[idxs+4*nz];
			opArray_re_s[5][is] = opArray_re[idxs+5*nz];

			opArray_im_s[0][is] = opArray_im[idxs];
			opArray_im_s[1][is] = opArray_im[idxs+1*nz];
			opArray_im_s[2][is] = opArray_im[idxs+2*nz];
			opArray_im_s[3][is] = opArray_im[idxs+3*nz];
			opArray_im_s[4][is] = opArray_im[idxs+4*nz];
			opArray_im_s[5][is] = opArray_im[idxs+5*nz];


		}

		for(is = 0; is < 3; is++)
		{
			idxs = (idx*blockSize) + is;

			// Construction and application of the nonlinear operator
			Nm_im = gamma*(powf(u[2][is+1],2)+ powf(u[3][is+1],2)+2*(powf(u[0][is+1],2)+powf(u[1][is+1],2)));
			Np_im = gamma*(powf(u[0][is-1],2)+ powf(u[1][is-1],2)+2*(powf(u[2][is-1],2)+powf(u[3][is-1],2)));
			um_nl_im = cosf(h*Nm_im)*u[3][is+1] + sinf(h*Nm_im)*u[2][is+1];
			um_nl_re = cosf(h*Nm_im)*u[2][is+1] - sinf(h*Nm_im)*u[3][is+1];
			up_nl_im = cosf(h*Np_im)*u[1][is-1] + sinf(h*Np_im)*u[0][is-1];
			up_nl_re = cosf(h*Np_im)*u[0][is-1] - sinf(h*Np_im)*u[1][is-1];

			// Application of the linear operator
			um_next_re = opArray_re_s[5][is]*(opArray_re_s[1][is]*um_nl_re - opArray_im_s[1][is]*um_nl_im) + opArray_re_s[5][is]*(opArray_re_s[2][is]*up_nl_re - opArray_im_s[2][is]*up_nl_im);
			um_next_im = opArray_re_s[5][is]*(opArray_re_s[1][is]*um_nl_im + opArray_im_s[1][is]*um_nl_re) + opArray_re_s[5][is]*(opArray_re_s[2][is]*up_nl_im + opArray_im_s[2][is]*up_nl_re);
			up_next_re = opArray_re_s[5][is]*(opArray_re_s[3][is]*um_nl_re - opArray_im_s[3][is]*um_nl_im) + opArray_re_s[5][is]*(opArray_re_s[4][is]*up_nl_re - opArray_im_s[4][is]*up_nl_im);
			up_next_im = opArray_re_s[5][is]*(opArray_re_s[3][is]*um_nl_im + opArray_im_s[3][is]*um_nl_re) + opArray_re_s[5][is]*(opArray_re_s[4][is]*up_nl_im + opArray_im_s[4][is]*up_nl_re);

			__syncthreads();

			up_re[idxs] = up_next_re;
			up_im[idxs] = up_next_im;
			um_re[idxs] = um_next_re;
			um_im[idxs] = um_next_im;

	 	}

	  }
	""")

	####### Optimized SplitStep Method Kernel end #######
	#####################################################


	func = mod.get_function("splitstep")
	for i in arange(nt):

			"""
			paramatersInt[0] = nt
			paramatersInt_gpu = cuda.mem_alloc(paramatersInt.size*paramatersInt.dtype.itemsize)
			cuda.memcpy_htod(paramatersInt_gpu, paramatersInt)
			"""

			func(up_gpu, up_im_gpu, um_gpu, um_im_gpu, opArray_gpu, opArray_im_gpu, timeDetector_gpu, timeDetector_im_gpu, block=dimBlock, grid=dimGrid)
			
			# Archive le signal avec une periode stepArchive
			if archiveFlag == stepArchive:
				cuda.memcpy_dtoh(upArchiveInt32[ia], up_gpu)
				cuda.memcpy_dtoh(umArchiveInt32[ia], um_gpu)
				cuda.memcpy_dtoh(up_im_ArchiveInt32[ia], up_im_gpu)
				cuda.memcpy_dtoh(um_im_ArchiveInt32[ia], um_im_gpu)
				upArchiveInt[ia] = pow(upArchiveInt32[ia],2) + pow(up_im_ArchiveInt32[ia],2)
				umArchiveInt[ia] = pow(umArchiveInt32[ia],2) + pow(um_im_ArchiveInt32[ia],2)
				ia += 1
				archiveFlag = 1
			else:
				archiveFlag += 1
			

 	# Recopy the result from the gpu memory to the main memory
	up_final32 = empty_like(up_float32)
	um_final32 = empty_like(um_float32)
	up_final32_im = empty_like(up_float32)
	um_final32_im = empty_like(um_float32)
	cuda.memcpy_dtoh(up_final32, up_gpu)
	cuda.memcpy_dtoh(um_final32, um_gpu)
	cuda.memcpy_dtoh(up_final32_im, up_im_gpu)
	cuda.memcpy_dtoh(um_final32_im, um_im_gpu)
	cuda.memcpy_dtoh(timeDetector32 , timeDetector_gpu)
	cuda.memcpy_dtoh(timeDetector32_im, timeDetector_im_gpu)

	##########################################

	# Convert to numpy complex type
	um_final = zeros(nz, complex)
	up_final = zeros(nz, complex)
	um_final.real = um_final32
	um_final.imag = um_final32_im
	up_final.real = up_final32
	up_final.imag = up_final32_im
	timeDetector.real = timeDetector32
	timeDetector.imag = timeDetector32_im

	return [up_final, um_final, upArchiveInt, umArchiveInt, timeDetector]


def ossmgpu2(up, length, nt, alpha, beta1, kappa, detuning, gamma, stepArchive = 1, dz_over_dt = 1.0, posDectector = 0):
	'''	
	Implementation of the optimized split-step method for solving nonlinear 
	coupled-mode equations that model wave propagation in nonlinear fiber Bragg
	gratings. (version nonsymmetrized)

	ref. Toroker et al. 'Optimized split-step method for modeling
	nonlinear pulse propagation in nonlinear fiber Bragg gratings'

	 -- Version CUDA in-place -- 

	up: 		Profil spatial de l'impulsion
	length:		Longueur du fbg
	nt:			Nombre de steps temporel
	alpha:		Parametre de gain
	beta1:		Constante de propagation (1/vitesse de groupe)
	kappa:		Parametre de couplage du fbg [1/m]
	detuning:	Detuning par rapport a la longueur d'onde de bragg [1/m]
	'''


	import pycuda.driver as cuda
	import pycuda.compiler as cudaComp
	import pycuda.autoinit

	# Vitesse de groupe dans le fibre sans fbg [m/ps]
	Vg = (1.0/beta1)/(1e12)
	nz = len(up)
	dz = length / nz
	k = wspace(dz*nz,nz)

	# Cas simple ou dt est fixee a dz par la vitesse de groupe [ps]
	dt = dz / Vg
	h = dz

	# On suppose que u_moins = 0 a z = L
	um = zeros(nz, complex)
	upArchiveInt = zeros([nt/stepArchive,nz], double)
	umArchiveInt = zeros([nt/stepArchive,nz], double)
	upArchiveInt32 = zeros([nt/stepArchive,nz], float32)
	umArchiveInt32 = zeros([nt/stepArchive,nz], float32)
	up_im_ArchiveInt32 = zeros([nt/stepArchive,nz], float32)
	um_im_ArchiveInt32 = zeros([nt/stepArchive,nz], float32)
	timeDetector = zeros(nt, complex)
	timeDetector32 = zeros(nt, float32)
	timeDetector32_im = zeros(nt, float32)

	# Matrice qui va contenir le stuff a envoyer au gpu
	operatorArray = zeros([6,nz], float32)
	operatorArray_im = zeros([6,nz], float32)

	# Construction de l'operateur lineaire
	Dmg = exp(h*(-alpha/2.0))
	Dm1 = (cos(detuning*h)+1.0j*sin(detuning*h))*cos(kappa*h)
	Dm2 = (cos(detuning*h)+1.0j*sin(detuning*h))*1.0j*sin(kappa*h)
	Dp1 = (cos(detuning*h)+1.0j*sin(detuning*h))*1.0j*sin(kappa*h)
	Dp2 = (cos(detuning*h)+1.0j*sin(detuning*h))*cos(kappa*h)

	# Transfert des operateurs et parametres
	operatorArray[0,0] = float(nz)
	operatorArray[0,1] = nt
	operatorArray[0,2] = gamma
	operatorArray[0,3] = h
	operatorArray[0,4] = posDectector

	operatorArray[1] = Dm1.real.astype(float32)
	operatorArray_im[1] = Dm1.imag.astype(float32)
	operatorArray[2] = Dm2.real.astype(float32)
	operatorArray_im[2] = Dm2.imag.astype(float32)
	operatorArray[3] = Dp1.real.astype(float32)
	operatorArray_im[3] = Dp1.imag.astype(float32)
	operatorArray[4] = Dp2.real.astype(float32)
	operatorArray_im[4] = Dp2.imag.astype(float32)
	operatorArray[5] = exp(h*(-alpha/2.0))

	paramatersInt = zeros(1, int)
	
	archiveFlag = stepArchive
	ia = 0

	##########################################
	### Initialisation of the PyCUDA Stuff ###
	##########################################
	
	dimBlock = (256,1,1)
	dimGrid	= (nz/dimBlock[0],1)

	# Convert to float32
	up_float32 = up.real.astype(float32)
	up_float32_im = up.imag.astype(float32)
	um_float32 = um.real.astype(float32)
	um_float32_im = um.imag.astype(float32)

	# Allocation and copy to the gpu memory
	opArray_gpu = cuda.mem_alloc(operatorArray.size*operatorArray.dtype.itemsize)
	cuda.memcpy_htod(opArray_gpu, operatorArray)
	opArray_im_gpu = cuda.mem_alloc(operatorArray_im.size*operatorArray_im.dtype.itemsize)
	cuda.memcpy_htod(opArray_im_gpu, operatorArray_im)

	timeDetector_gpu = cuda.mem_alloc(timeDetector32.size*timeDetector32.dtype.itemsize)
	cuda.memcpy_htod(timeDetector_gpu, timeDetector32)
	timeDetector_im_gpu = cuda.mem_alloc(timeDetector32_im.size*timeDetector32_im.dtype.itemsize)
	cuda.memcpy_htod(timeDetector_im_gpu, timeDetector32_im)

	up_gpu = cuda.mem_alloc(up_float32.size*up_float32.dtype.itemsize)
	cuda.memcpy_htod(up_gpu, up_float32)
	up_im_gpu = cuda.mem_alloc(up_float32_im.size*up_float32_im.dtype.itemsize)
	cuda.memcpy_htod(up_im_gpu, up_float32_im)

	um_gpu = cuda.mem_alloc(um_float32.size*um_float32.dtype.itemsize)
	cuda.memcpy_htod(um_gpu, um_float32)
	um_im_gpu = cuda.mem_alloc(um_float32_im.size*um_float32_im.dtype.itemsize)
	cuda.memcpy_htod(um_im_gpu, um_float32_im)


	######### Optimized SplitStep Method Kernel	#########
	#####################################################

	mod = cudaComp.SourceModule("""

	  __global__ void splitstep(float *up_re, float *up_im, float *um_re, float *um_im, float *opArray_re, float *opArray_im, float *timeDetector, float *timeDetector_im)
	  {

		float up_next_im;
		float up_next_re;
		float um_next_im;
		float um_next_re;

		const uint idx = threadIdx.x + (blockIdx.x * blockDim.x);

		// Get the simulation parameters store in the unused part of opArray_re
		const uint nz = (int)opArray_re[0];
		//const uint nt = (int)opArray_re[1];
		const float gamma = opArray_re[2];
		const float h = opArray_re[3];
		//const uint posDetector = (int)opArray_re[4];

		// Load stuff into the registers
		// up_re = u[0], up_im = u[1], um_re = u[2], um_im = u[3]
		// up_re[idx-1] = up[0][0], up_re[idx] = up[0][1], up_re[idx+1] = up[0][2] , etc...
		float u[4][3];
		//float op[5][2];

		// Device memory to the register
		// Make sure that the concatenated value are zero
		if(idx <= 0)
		{
			u[0][0] = 0.0;
			u[1][0] = 0.0;
			u[2][0] = 0.0;
			u[3][0] = 0.0;
		}
		else
		{
			u[0][0] =  up_re[idx-1];
			u[1][0] =  up_im[idx-1];
			u[2][0] =  um_re[idx-1];
			u[3][0] =  um_im[idx-1];
		}

		if(idx >= (nz-1))
		{
			u[0][2] = 0.0;
			u[1][2] = 0.0;
			u[2][2] = 0.0;
			u[3][2] = 0.0;
		}
		else
		{
			u[0][2] = up_re[idx+1];
			u[1][2] = up_im[idx+1];
			u[2][2] = um_re[idx+1];
			u[3][2] = um_im[idx+1];
		}

		u[0][1] =  up_re[idx];
		u[1][1] =  up_im[idx];
		u[2][1] =  um_re[idx];
		u[3][1] =  um_im[idx];

		
		/* Global to register
		op[0][0] = opArray_re[idx+5*nz];
		op[1][0] = opArray_re[idx+1*nz];
		op[1][1] = opArray_im[idx+1*nz];
		op[2][0] = opArray_re[idx+2*nz];
		op[2][1] = opArray_im[idx+2*nz];
		op[3][0] = opArray_re[idx+3*nz];
		op[3][1] = opArray_im[idx+3*nz];
		op[4][0] = opArray_re[idx+4*nz];
		op[4][1] = opArray_im[idx+4*nz];
		*/


		// Construction and application of the nonlinear operator
		// Use the __fmul_rn() to avoid the combination into a FMAD by the compiler
		const float Nm_im = __fmul_rn(gamma,(powf(u[2][2],2.0)+powf(u[3][2],2.0) + __fmul_rn(2.0,(powf(u[0][2],2.0)+powf(u[1][2],2.0)))));
		const float Np_im = __fmul_rn(gamma,(powf(u[0][0],2.0)+powf(u[1][0],2.0) + __fmul_rn(2.0,(powf(u[2][0],2.0)+powf(u[3][0],2.0)))));
		const float um_nl_im = __fmul_rn(cosf(h*Nm_im),u[3][2]) + __fmul_rn(sinf(h*Nm_im),u[2][2]);
		const float um_nl_re = __fmul_rn(cosf(h*Nm_im),u[2][2]) - __fmul_rn(sinf(h*Nm_im),u[3][2]);
		const float up_nl_im = __fmul_rn(cosf(h*Np_im),u[1][0]) + __fmul_rn(sinf(h*Np_im),u[0][0]);
		const float up_nl_re = __fmul_rn(cosf(h*Np_im),u[0][0]) - __fmul_rn(sinf(h*Np_im),u[1][0]);

		/*
		// Application des operateurs lineaires - Version register
		um_next_re = op[0][0]*(op[1][0]*um_nl_re - op[1][1]*um_nl_im) + op[0][0]*(op[2][0]*up_nl_re - op[2][1]*up_nl_im);
		um_next_im = op[0][0]*(op[1][0]*um_nl_im + op[1][1]*um_nl_re) + op[0][0]*(op[2][0]*up_nl_im + op[2][1]*up_nl_re);
		up_next_re = op[0][0]*(op[3][0]*um_nl_re - op[3][1]*um_nl_im) + op[0][0]*(op[4][0]*up_nl_re - op[4][1]*up_nl_im);
		up_next_im = op[0][0]*(op[3][0]*um_nl_im + op[3][1]*um_nl_re) + op[0][0]*(op[4][0]*up_nl_im + op[4][1]*up_nl_re);
		*/

		/*
		// Application of the linear operator - Version global
		um_next_re = opArray_re[idx+5*nz]*(opArray_re[idx+1*nz]*um_nl_re - opArray_im[idx+1*nz]*um_nl_im) + opArray_re[idx+5*nz]*(opArray_re[idx+2*nz]*up_nl_re - opArray_im[idx+2*nz]*up_nl_im);
		um_next_im = opArray_re[idx+5*nz]*(opArray_re[idx+1*nz]*um_nl_im + opArray_im[idx+1*nz]*um_nl_re) + opArray_re[idx+5*nz]*(opArray_re[idx+2*nz]*up_nl_im + opArray_im[idx+2*nz]*up_nl_re);
		up_next_re = opArray_re[idx+5*nz]*(opArray_re[idx+3*nz]*um_nl_re - opArray_im[idx+3*nz]*um_nl_im) + opArray_re[idx+5*nz]*(opArray_re[idx+4*nz]*up_nl_re - opArray_im[idx+4*nz]*up_nl_im);
		up_next_im = opArray_re[idx+5*nz]*(opArray_re[idx+3*nz]*um_nl_im + opArray_im[idx+3*nz]*um_nl_re) + opArray_re[idx+5*nz]*(opArray_re[idx+4*nz]*up_nl_im + opArray_im[idx+4*nz]*up_nl_re);
		*/

		
		// Application of the linear operator - Version global
		um_next_re = __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+1*nz],um_nl_re) - __fmul_rn(opArray_im[idx+1*nz],um_nl_im)));
		um_next_re += __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+2*nz],up_nl_re) - __fmul_rn(opArray_im[idx+2*nz],up_nl_im)));
		um_next_im = __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+1*nz],um_nl_im) + __fmul_rn(opArray_im[idx+1*nz],um_nl_re)));
		um_next_im += __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+2*nz],up_nl_im) + __fmul_rn(opArray_im[idx+2*nz],up_nl_re)));
		up_next_re = __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+3*nz],um_nl_re) - __fmul_rn(opArray_im[idx+3*nz],um_nl_im)));
		up_next_re += __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+4*nz],up_nl_re) - __fmul_rn(opArray_im[idx+4*nz],up_nl_im)));
		up_next_im = __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+3*nz],um_nl_im) + __fmul_rn(opArray_im[idx+3*nz],um_nl_re)));
		up_next_im += __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+4*nz],up_nl_im) + __fmul_rn(opArray_im[idx+4*nz],up_nl_re)));

		__syncthreads();

		up_re[idx] = up_next_re;
		up_im[idx] = up_next_im;
		um_re[idx] = um_next_re;
		um_im[idx] = um_next_im;
		
		//timeDetector[nt] = up_re[posDetector];
		//timeDetector_im[nt] = up_im[posDetector];
		
	  }

	""")

	####### Optimized SplitStep Method Kernel end #######
	#####################################################


	func = mod.get_function("splitstep")
	for i in arange(nt):

			"""
			paramatersInt[0] = nt
			paramatersInt_gpu = cuda.mem_alloc(paramatersInt.size*paramatersInt.dtype.itemsize)
			cuda.memcpy_htod(paramatersInt_gpu, paramatersInt)
			"""

			func(up_gpu, up_im_gpu, um_gpu, um_im_gpu, opArray_gpu, opArray_im_gpu, timeDetector_gpu, timeDetector_im_gpu, block=dimBlock, grid=dimGrid)
		
			# Archive le signal avec une periode stepArchive
			if archiveFlag == stepArchive:
				cuda.memcpy_dtoh(upArchiveInt32[ia], up_gpu)
				cuda.memcpy_dtoh(umArchiveInt32[ia], um_gpu)
				cuda.memcpy_dtoh(up_im_ArchiveInt32[ia], up_im_gpu)
				cuda.memcpy_dtoh(um_im_ArchiveInt32[ia], um_im_gpu)
				upArchiveInt[ia] = pow(upArchiveInt32[ia],2) + pow(up_im_ArchiveInt32[ia],2)
				umArchiveInt[ia] = pow(umArchiveInt32[ia],2) + pow(um_im_ArchiveInt32[ia],2)
				ia += 1
				archiveFlag = 1
			else:
				archiveFlag += 1
			

 	# Recopy the result from the gpu memory to the main memory
	up_final32 = empty_like(up_float32)
	um_final32 = empty_like(um_float32)
	up_final32_im = empty_like(up_float32)
	um_final32_im = empty_like(um_float32)
	cuda.memcpy_dtoh(up_final32, up_gpu)
	cuda.memcpy_dtoh(um_final32, um_gpu)
	cuda.memcpy_dtoh(up_final32_im, up_im_gpu)
	cuda.memcpy_dtoh(um_final32_im, um_im_gpu)
	cuda.memcpy_dtoh(timeDetector32 , timeDetector_gpu)
	cuda.memcpy_dtoh(timeDetector32_im, timeDetector_im_gpu)

	##########################################

	# Convert to numpy complex type
	um_final = zeros(nz, complex)
	up_final = zeros(nz, complex)
	um_final.real = um_final32
	um_final.imag = um_final32_im
	up_final.real = up_final32
	up_final.imag = up_final32_im
	timeDetector.real = timeDetector32
	timeDetector.imag = timeDetector32_im

	return [up_final, um_final, upArchiveInt, umArchiveInt, timeDetector]


def ossmgpu2Amp(up, length, nt, alpha, beta1, kappa, detuning, gamma, nbrArchive = 1, dz_over_dt = 1.0, posDectector = 0):
	'''	
	Implementation of the optimized split-step method for solving nonlinear 
	coupled-mode equations that model wave propagation in nonlinear fiber Bragg
	gratings. (version nonsymmetrized)

	ref. Toroker et al. 'Optimized split-step method for modeling
	nonlinear pulse propagation in nonlinear fiber Bragg gratings'

	 -- Version CUDA in-place -- 

	up: 		Profil spatial de l'impulsion
	length:		Longueur du fbg
	nt:			Nombre de steps temporel
	alpha:		Parametre de gain
	beta1:		Constante de propagation (1/vitesse de groupe)
	kappa:		Parametre de couplage du fbg [1/m]
	detuning:	Detuning par rapport a la longueur d'onde de bragg [1/m]
	'''


	import pycuda.driver as cuda
	import pycuda.compiler as cudaComp
	import pycuda.autoinit

	# Vitesse de groupe dans le fibre sans fbg [m/ps]
	Vg = (1.0/beta1)/(1e12)
	nz = len(up)
	dz = length / nz
	k = wspace(dz*nz,nz)

	# Cas simple ou dt est fixee a dz par la vitesse de groupe [ps]
	dt = dz / Vg
	h = dz

	# On suppose que u_moins = 0 a z = L
	um = zeros(nz, complex)
	upArchiveAmp = zeros([nbrArchive,nz], complex)
	umArchiveAmp = zeros([nbrArchive,nz], complex)
	upArchiveInt32 = zeros([nbrArchive,nz], float32)
	umArchiveInt32 = zeros([nbrArchive,nz], float32)
	up_im_ArchiveInt32 = zeros([nbrArchive,nz], float32)
	um_im_ArchiveInt32 = zeros([nbrArchive,nz], float32)
	timeDetector = zeros(nt, complex)
	timeDetector32 = zeros(nt, float32)
	timeDetector32_im = zeros(nt, float32)

	# Matrice qui va contenir le stuff a envoyer au gpu
	operatorArray = zeros([6,nz], float32)
	operatorArray_im = zeros([6,nz], float32)

	# Construction de l'operateur lineaire
	Dmg = exp(h*(-alpha/2.0))
	Dm1 = (cos(detuning*h)+1.0j*sin(detuning*h))*cos(kappa*h)
	Dm2 = (cos(detuning*h)+1.0j*sin(detuning*h))*1.0j*sin(kappa*h)
	Dp1 = (cos(detuning*h)+1.0j*sin(detuning*h))*1.0j*sin(kappa*h)
	Dp2 = (cos(detuning*h)+1.0j*sin(detuning*h))*cos(kappa*h)

	# Transfert des operateurs et parametres
	operatorArray[0,0] = float(nz)
	operatorArray[0,1] = nt
	operatorArray[0,2] = gamma
	operatorArray[0,3] = h
	operatorArray[0,4] = posDectector

	operatorArray[1] = Dm1.real.astype(float32)
	operatorArray_im[1] = Dm1.imag.astype(float32)
	operatorArray[2] = Dm2.real.astype(float32)
	operatorArray_im[2] = Dm2.imag.astype(float32)
	operatorArray[3] = Dp1.real.astype(float32)
	operatorArray_im[3] = Dp1.imag.astype(float32)
	operatorArray[4] = Dp2.real.astype(float32)
	operatorArray_im[4] = Dp2.imag.astype(float32)
	operatorArray[5] = exp(h*(-alpha/2.0))

	paramatersInt = zeros(1, int)
	
	archArray = linspace(0,nt-1,nbrArchive).astype(int)
	ia = 0


	##########################################
	### Initialisation of the PyCUDA Stuff ###
	##########################################
	
	dimBlock = (256,1,1)
	dimGrid	= (nz/dimBlock[0],1)

	# Convert to float32
	up_float32 = up.real.astype(float32)
	up_float32_im = up.imag.astype(float32)
	um_float32 = um.real.astype(float32)
	um_float32_im = um.imag.astype(float32)

	# Allocation and copy to the gpu memory
	opArray_gpu = cuda.mem_alloc(operatorArray.size*operatorArray.dtype.itemsize)
	cuda.memcpy_htod(opArray_gpu, operatorArray)
	opArray_im_gpu = cuda.mem_alloc(operatorArray_im.size*operatorArray_im.dtype.itemsize)
	cuda.memcpy_htod(opArray_im_gpu, operatorArray_im)

	timeDetector_gpu = cuda.mem_alloc(timeDetector32.size*timeDetector32.dtype.itemsize)
	cuda.memcpy_htod(timeDetector_gpu, timeDetector32)
	timeDetector_im_gpu = cuda.mem_alloc(timeDetector32_im.size*timeDetector32_im.dtype.itemsize)
	cuda.memcpy_htod(timeDetector_im_gpu, timeDetector32_im)

	up_gpu = cuda.mem_alloc(up_float32.size*up_float32.dtype.itemsize)
	cuda.memcpy_htod(up_gpu, up_float32)
	up_im_gpu = cuda.mem_alloc(up_float32_im.size*up_float32_im.dtype.itemsize)
	cuda.memcpy_htod(up_im_gpu, up_float32_im)

	um_gpu = cuda.mem_alloc(um_float32.size*um_float32.dtype.itemsize)
	cuda.memcpy_htod(um_gpu, um_float32)
	um_im_gpu = cuda.mem_alloc(um_float32_im.size*um_float32_im.dtype.itemsize)
	cuda.memcpy_htod(um_im_gpu, um_float32_im)


	######### Optimized SplitStep Method Kernel	#########
	#####################################################

	mod = cudaComp.SourceModule("""

	  __global__ void splitstep(float *up_re, float *up_im, float *um_re, float *um_im, float *opArray_re, float *opArray_im, float *timeDetector, float *timeDetector_im)
	  {

		float up_next_im;
		float up_next_re;
		float um_next_im;
		float um_next_re;

		const uint idx = threadIdx.x + (blockIdx.x * blockDim.x);

		// Get the simulation parameters store in the unused part of opArray_re
		const uint nz = (int)opArray_re[0];
		//const uint nt = (int)opArray_re[1];
		const float gamma = opArray_re[2];
		const float h = opArray_re[3];
		//const uint posDetector = (int)opArray_re[4];

		// Load stuff into the registers
		// up_re = u[0], up_im = u[1], um_re = u[2], um_im = u[3]
		// up_re[idx-1] = up[0][0], up_re[idx] = up[0][1], up_re[idx+1] = up[0][2] , etc...
		float u[4][3];
		//float op[5][2];

		// Device memory to the register
		// Make sure that the concatenated value are zero
		if(idx <= 0)
		{
			u[0][0] = 0.0;
			u[1][0] = 0.0;
			u[2][0] = 0.0;
			u[3][0] = 0.0;
		}
		else
		{
			u[0][0] =  up_re[idx-1];
			u[1][0] =  up_im[idx-1];
			u[2][0] =  um_re[idx-1];
			u[3][0] =  um_im[idx-1];
		}

		if(idx >= (nz-1))
		{
			u[0][2] = 0.0;
			u[1][2] = 0.0;
			u[2][2] = 0.0;
			u[3][2] = 0.0;
		}
		else
		{
			u[0][2] = up_re[idx+1];
			u[1][2] = up_im[idx+1];
			u[2][2] = um_re[idx+1];
			u[3][2] = um_im[idx+1];
		}

		u[0][1] =  up_re[idx];
		u[1][1] =  up_im[idx];
		u[2][1] =  um_re[idx];
		u[3][1] =  um_im[idx];

		
		/* Global to register
		op[0][0] = opArray_re[idx+5*nz];
		op[1][0] = opArray_re[idx+1*nz];
		op[1][1] = opArray_im[idx+1*nz];
		op[2][0] = opArray_re[idx+2*nz];
		op[2][1] = opArray_im[idx+2*nz];
		op[3][0] = opArray_re[idx+3*nz];
		op[3][1] = opArray_im[idx+3*nz];
		op[4][0] = opArray_re[idx+4*nz];
		op[4][1] = opArray_im[idx+4*nz];
		*/


		// Construction and application of the nonlinear operator
		// Use the __fmul_rn() to avoid the combination into a FMAD by the compiler
		const float Nm_im = __fmul_rn(gamma,(powf(u[2][2],2.0)+powf(u[3][2],2.0) + __fmul_rn(2.0,(powf(u[0][2],2.0)+powf(u[1][2],2.0)))));
		const float Np_im = __fmul_rn(gamma,(powf(u[0][0],2.0)+powf(u[1][0],2.0) + __fmul_rn(2.0,(powf(u[2][0],2.0)+powf(u[3][0],2.0)))));
		const float um_nl_im = __fmul_rn(cosf(h*Nm_im),u[3][2]) + __fmul_rn(sinf(h*Nm_im),u[2][2]);
		const float um_nl_re = __fmul_rn(cosf(h*Nm_im),u[2][2]) - __fmul_rn(sinf(h*Nm_im),u[3][2]);
		const float up_nl_im = __fmul_rn(cosf(h*Np_im),u[1][0]) + __fmul_rn(sinf(h*Np_im),u[0][0]);
		const float up_nl_re = __fmul_rn(cosf(h*Np_im),u[0][0]) - __fmul_rn(sinf(h*Np_im),u[1][0]);

		/*
		// Application des operateurs lineaires - Version register
		um_next_re = op[0][0]*(op[1][0]*um_nl_re - op[1][1]*um_nl_im) + op[0][0]*(op[2][0]*up_nl_re - op[2][1]*up_nl_im);
		um_next_im = op[0][0]*(op[1][0]*um_nl_im + op[1][1]*um_nl_re) + op[0][0]*(op[2][0]*up_nl_im + op[2][1]*up_nl_re);
		up_next_re = op[0][0]*(op[3][0]*um_nl_re - op[3][1]*um_nl_im) + op[0][0]*(op[4][0]*up_nl_re - op[4][1]*up_nl_im);
		up_next_im = op[0][0]*(op[3][0]*um_nl_im + op[3][1]*um_nl_re) + op[0][0]*(op[4][0]*up_nl_im + op[4][1]*up_nl_re);
		*/

		/*
		// Application of the linear operator - Version global
		um_next_re = opArray_re[idx+5*nz]*(opArray_re[idx+1*nz]*um_nl_re - opArray_im[idx+1*nz]*um_nl_im) + opArray_re[idx+5*nz]*(opArray_re[idx+2*nz]*up_nl_re - opArray_im[idx+2*nz]*up_nl_im);
		um_next_im = opArray_re[idx+5*nz]*(opArray_re[idx+1*nz]*um_nl_im + opArray_im[idx+1*nz]*um_nl_re) + opArray_re[idx+5*nz]*(opArray_re[idx+2*nz]*up_nl_im + opArray_im[idx+2*nz]*up_nl_re);
		up_next_re = opArray_re[idx+5*nz]*(opArray_re[idx+3*nz]*um_nl_re - opArray_im[idx+3*nz]*um_nl_im) + opArray_re[idx+5*nz]*(opArray_re[idx+4*nz]*up_nl_re - opArray_im[idx+4*nz]*up_nl_im);
		up_next_im = opArray_re[idx+5*nz]*(opArray_re[idx+3*nz]*um_nl_im + opArray_im[idx+3*nz]*um_nl_re) + opArray_re[idx+5*nz]*(opArray_re[idx+4*nz]*up_nl_im + opArray_im[idx+4*nz]*up_nl_re);
		*/

		
		// Application of the linear operator - Version global
		um_next_re = __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+1*nz],um_nl_re) - __fmul_rn(opArray_im[idx+1*nz],um_nl_im)));
		um_next_re += __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+2*nz],up_nl_re) - __fmul_rn(opArray_im[idx+2*nz],up_nl_im)));
		um_next_im = __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+1*nz],um_nl_im) + __fmul_rn(opArray_im[idx+1*nz],um_nl_re)));
		um_next_im += __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+2*nz],up_nl_im) + __fmul_rn(opArray_im[idx+2*nz],up_nl_re)));
		up_next_re = __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+3*nz],um_nl_re) - __fmul_rn(opArray_im[idx+3*nz],um_nl_im)));
		up_next_re += __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+4*nz],up_nl_re) - __fmul_rn(opArray_im[idx+4*nz],up_nl_im)));
		up_next_im = __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+3*nz],um_nl_im) + __fmul_rn(opArray_im[idx+3*nz],um_nl_re)));
		up_next_im += __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+4*nz],up_nl_im) + __fmul_rn(opArray_im[idx+4*nz],up_nl_re)));

		__syncthreads();

		up_re[idx] = up_next_re;
		up_im[idx] = up_next_im;
		um_re[idx] = um_next_re;
		um_im[idx] = um_next_im;
		
		//timeDetector[nt] = up_re[posDetector];
		//timeDetector_im[nt] = up_im[posDetector];
		
	  }

	""")

	####### Optimized SplitStep Method Kernel end #######
	#####################################################


	func = mod.get_function("splitstep")
	for i in arange(nt):

			"""
			paramatersInt[0] = nt
			paramatersInt_gpu = cuda.mem_alloc(paramatersInt.size*paramatersInt.dtype.itemsize)
			cuda.memcpy_htod(paramatersInt_gpu, paramatersInt)
			"""

			func(up_gpu, up_im_gpu, um_gpu, um_im_gpu, opArray_gpu, opArray_im_gpu, timeDetector_gpu, timeDetector_im_gpu, block=dimBlock, grid=dimGrid)
		
			# Archive le signal nbrArchive fois
			if i == archArray[ia]:
				cuda.memcpy_dtoh(upArchiveInt32[ia], up_gpu)
				cuda.memcpy_dtoh(umArchiveInt32[ia], um_gpu)
				cuda.memcpy_dtoh(up_im_ArchiveInt32[ia], up_im_gpu)
				cuda.memcpy_dtoh(um_im_ArchiveInt32[ia], um_im_gpu)
				upArchiveAmp[ia] = upArchiveInt32[ia] + 1.0j*up_im_ArchiveInt32[ia]
				umArchiveAmp[ia] = umArchiveInt32[ia] + 1.0j*um_im_ArchiveInt32[ia]
				print "Archive: " + str(i)
				ia += 1
				

 	# Recopy the result from the gpu memory to the main memory
	up_final32 = empty_like(up_float32)
	um_final32 = empty_like(um_float32)
	up_final32_im = empty_like(up_float32)
	um_final32_im = empty_like(um_float32)
	cuda.memcpy_dtoh(up_final32, up_gpu)
	cuda.memcpy_dtoh(um_final32, um_gpu)
	cuda.memcpy_dtoh(up_final32_im, up_im_gpu)
	cuda.memcpy_dtoh(um_final32_im, um_im_gpu)
	cuda.memcpy_dtoh(timeDetector32 , timeDetector_gpu)
	cuda.memcpy_dtoh(timeDetector32_im, timeDetector_im_gpu)

	##########################################

	# Convert to numpy complex type
	um_final = zeros(nz, complex)
	up_final = zeros(nz, complex)
	um_final.real = um_final32
	um_final.imag = um_final32_im
	up_final.real = up_final32
	up_final.imag = up_final32_im
	timeDetector.real = timeDetector32
	timeDetector.imag = timeDetector32_im

	return [upArchiveAmp, umArchiveAmp, archArray]



def testgpu(u):

	import pycuda.driver as cuda
	import pycuda.compiler as cudaComp
	import pycuda.autoinit

	size = len(u)
	N = 1

	dimBlock = (112,1,1)
	dimGrid	= (size/dimBlock[0],1)

	# kernel output
	u_out_float32 = zeros(size, float32)

	# Convert to float32
	u_float32 = u.real.astype(float32)

	u_gpu = cuda.mem_alloc(u_float32.size*u_float32.dtype.itemsize)
	cuda.memcpy_htod(u_gpu, u_float32)

	u_out_gpu = cuda.mem_alloc(u_out_float32.size*u_out_float32.dtype.itemsize)
	cuda.memcpy_htod(u_out_gpu, u_out_float32)

	mod = cudaComp.SourceModule("""

	__global__ void testKernel(float *u, float *u_out)
	{
			const uint idx = threadIdx.x + (blockIdx.x * blockDim.x);
		
			u_out[idx] = u[idx]*u[idx];
	}

	""")
	
	func = mod.get_function("testKernel")
	for n in arange(N):
		func(u_gpu, u_out_gpu, block=dimBlock, grid=dimGrid)

	cuda.memcpy_dtoh(u_out_float32, u_out_gpu)

	return u_out_float32


'''
def testgpu_simple(u):

	import pycuda.driver as cuda
	import pycuda.compiler as cudaComp
	import pycuda.autoinit

	size = len(u)
	N = 1

	dimBlock = (112,1,1)
	dimGrid	= (size/dimBlock[0],1)

	# kernel output
	u_out_float32 = zeros(size, float32)

	# Convert to float32
	u_float32 = u.real.astype(float32)

	u_gpu = cuda.mem_alloc(u_float32.size*u_float32.dtype.itemsize)
	cuda.memcpy_htod(u_gpu, u_float32)

	u_out_gpu = cuda.mem_alloc(u_out_float32.size*u_out_float32.dtype.itemsize)
	cuda.memcpy_htod(u_out_gpu, u_out_float32)

	mod = cudaComp.SourceModule("""

	__global__ void testKernel(float *u, float *u_out)
	{
			texture<float, 1, cudaReadModeElementType> u_tex;
			const uint idx = threadIdx.x + (blockIdx.x * blockDim.x);
		
			u_out[idx] = u[idx]*u[idx];
	}

	""")
	
	func = mod.get_function("testKernel")
    u_tex = mod.get_texref("u_tex")
	for n in arange(N):
		func(u_gpu, u_out_gpu, block=dimBlock, grid=dimGrid)

	cuda.memcpy_dtoh(u_out_float32, u_out_gpu)

	return u_out_float32
'''


def ossmgpu3(up, length, nt, alpha, beta1, kappa, detuning, gamma, stepArchive = 1, dz_over_dt = 1.0, posDectector = 0):
	'''	
	Implementation of the optimized split-step method for solving nonlinear 
	coupled-mode equations that model wave propagation in nonlinear fiber Bragg
	gratings. (version nonsymmetrized)

	ref. Toroker et al. 'Optimized split-step method for modeling
	nonlinear pulse propagation in nonlinear fiber Bragg gratings'

	 -- Version CUDA -- 

	up: 		Profil spatial de l'impulsion
	length:		Longueur du fbg
	nt:			Nombre de steps temporel
	alpha:		Parametre de gain
	beta1:		Constante de propagation (1/vitesse de groupe)
	kappa:		Parametre de couplage du fbg [1/m]
	detuning:	Detuning par rapport a la longueur d'onde de bragg [1/m]
	'''


	import pycuda.driver as cuda
	import pycuda.compiler as cudaComp
	import pycuda.autoinit

	# Vitesse de groupe dans le fibre sans fbg [m/ps]
	Vg = (1.0/beta1)/(1e12)
	nz = len(up)
	dz = length / nz
	k = wspace(dz*nz,nz)

	# Cas simple ou dt est fixee a dz par la vitesse de groupe [ps]
	dt = dz / Vg
	h = dz

	# On suppose que u_moins = 0 a z = L
	um = zeros(nz, complex)
	um_o = zeros(nz, complex)
	upArchiveInt = zeros([nt/stepArchive,nz], double)
	umArchiveInt = zeros([nt/stepArchive,nz], double)
	upArchiveInt32 = zeros([nt/stepArchive,nz], float32)
	umArchiveInt32 = zeros([nt/stepArchive,nz], float32)
	up_im_ArchiveInt32 = zeros([nt/stepArchive,nz], float32)
	um_im_ArchiveInt32 = zeros([nt/stepArchive,nz], float32)
	timeDetector = zeros(nt, complex)
	timeDetector32 = zeros(nt, float32)
	timeDetector32_im = zeros(nt, float32)

	# Matrice qui va contenir le stuff a envoyer au gpu
	operatorArray = zeros([6,nz], float32)
	operatorArray_im = zeros([6,nz], float32)

	# Construction de l'operateur lineaire
	Dmg = exp(h*(-alpha/2.0))
	Dm1 = (cos(detuning*h)+1.0j*sin(detuning*h))*cos(kappa*h)
	Dm2 = (cos(detuning*h)+1.0j*sin(detuning*h))*1.0j*sin(kappa*h)
	Dp1 = (cos(detuning*h)+1.0j*sin(detuning*h))*1.0j*sin(kappa*h)
	Dp2 = (cos(detuning*h)+1.0j*sin(detuning*h))*cos(kappa*h)

	# Transfert des operateurs et parametres
	operatorArray[0,0] = float(nz)
	operatorArray[0,1] = nt
	operatorArray[0,2] = gamma
	operatorArray[0,3] = h
	operatorArray[0,4] = posDectector

	operatorArray[1] = Dm1.real.astype(float32)
	operatorArray_im[1] = Dm1.imag.astype(float32)
	operatorArray[2] = Dm2.real.astype(float32)
	operatorArray_im[2] = Dm2.imag.astype(float32)
	operatorArray[3] = Dp1.real.astype(float32)
	operatorArray_im[3] = Dp1.imag.astype(float32)
	operatorArray[4] = Dp2.real.astype(float32)
	operatorArray_im[4] = Dp2.imag.astype(float32)
	operatorArray[5] = exp(h*(-alpha/2.0))

	paramatersInt = zeros(1, int)
	
	archiveFlag = stepArchive
	ia = 0

	##########################################
	### Initialisation of the PyCUDA Stuff ###
	##########################################
	
	dimBlock = (256,1,1)
	dimGrid	= (nz/dimBlock[0],1)

	# Convert to float32
	up_float32 = up.real.astype(float32)
	up_float32_im = up.imag.astype(float32)
	um_float32 = um.real.astype(float32)
	um_float32_im = um.imag.astype(float32)

	# Allocation and copy to the gpu memory
	opArray_gpu = cuda.mem_alloc(operatorArray.size*operatorArray.dtype.itemsize)
	cuda.memcpy_htod(opArray_gpu, operatorArray)
	opArray_im_gpu = cuda.mem_alloc(operatorArray_im.size*operatorArray_im.dtype.itemsize)
	cuda.memcpy_htod(opArray_im_gpu, operatorArray_im)

	timeDetector_gpu = cuda.mem_alloc(timeDetector32.size*timeDetector32.dtype.itemsize)
	cuda.memcpy_htod(timeDetector_gpu, timeDetector32)
	timeDetector_im_gpu = cuda.mem_alloc(timeDetector32_im.size*timeDetector32_im.dtype.itemsize)
	cuda.memcpy_htod(timeDetector_im_gpu, timeDetector32_im)

	up_gpu = cuda.mem_alloc(up_float32.size*up_float32.dtype.itemsize)
	cuda.memcpy_htod(up_gpu, up_float32)
	up_im_gpu = cuda.mem_alloc(up_float32_im.size*up_float32_im.dtype.itemsize)
	cuda.memcpy_htod(up_im_gpu, up_float32_im)
	um_gpu = cuda.mem_alloc(um_float32.size*um_float32.dtype.itemsize)
	cuda.memcpy_htod(um_gpu, um_float32)
	um_im_gpu = cuda.mem_alloc(um_float32_im.size*um_float32_im.dtype.itemsize)
	cuda.memcpy_htod(um_im_gpu, um_float32_im)

	# Allocate array for the output
	up_gpu_o = cuda.mem_alloc(up_float32.size*up_float32.dtype.itemsize)
	up_im_gpu_o = cuda.mem_alloc(up_float32_im.size*up_float32_im.dtype.itemsize)
	um_gpu_o = cuda.mem_alloc(um_float32.size*um_float32.dtype.itemsize)
	um_im_gpu_o = cuda.mem_alloc(um_float32_im.size*um_float32_im.dtype.itemsize)


	######### Optimized SplitStep Method Kernel	#########
	#####################################################

	mod = cudaComp.SourceModule("""

	  __global__ void splitstep(float *up_re,
								float *up_im,
								float *um_re,
								float *um_im,
								float *up_re_o,
								float *up_im_o,
								float *um_re_o,
								float *um_im_o,
								float *opArray_re,
								float *opArray_im,
								float *timeDetector,
								float *timeDetector_im)
	  {

		float up_next_im;
		float up_next_re;
		float um_next_im;
		float um_next_re;

		const uint idx = threadIdx.x + (blockIdx.x * blockDim.x);

		// Get the simulation parameters store in the unused part of opArray_re
		const uint nz = (int)opArray_re[0];
		//const uint nt = (int)opArray_re[1];
		const float gamma = opArray_re[2];
		const float h = opArray_re[3];
		//const uint posDetector = (int)opArray_re[4];

		// Load stuff into the registers
		// up_re = u[0], up_im = u[1], um_re = u[2], um_im = u[3]
		// up_re[idx-1] = up[0][0], up_re[idx] = up[0][1], up_re[idx+1] = up[0][2] , etc...
		float u[4][3];
		//float op[5][2];

		// Device memory to the register
		// Make sure that the concatenated value are zero
		if(idx <= 0)
		{
			u[0][0] = 0.0;
			u[1][0] = 0.0;
			u[2][0] = 0.0;
			u[3][0] = 0.0;
		}
		else
		{
			u[0][0] =  up_re[idx-1];
			u[1][0] =  up_im[idx-1];
			u[2][0] =  um_re[idx-1];
			u[3][0] =  um_im[idx-1];
		}

		if(idx >= (nz-1))
		{
			u[0][2] = 0.0;
			u[1][2] = 0.0;
			u[2][2] = 0.0;
			u[3][2] = 0.0;
		}
		else
		{
			u[0][2] = up_re[idx+1];
			u[1][2] = up_im[idx+1];
			u[2][2] = um_re[idx+1];
			u[3][2] = um_im[idx+1];
		}

		u[0][1] =  up_re[idx];
		u[1][1] =  up_im[idx];
		u[2][1] =  um_re[idx];
		u[3][1] =  um_im[idx];

		
		/* Global to register
		op[0][0] = opArray_re[idx+5*nz];
		op[1][0] = opArray_re[idx+1*nz];
		op[1][1] = opArray_im[idx+1*nz];
		op[2][0] = opArray_re[idx+2*nz];
		op[2][1] = opArray_im[idx+2*nz];
		op[3][0] = opArray_re[idx+3*nz];
		op[3][1] = opArray_im[idx+3*nz];
		op[4][0] = opArray_re[idx+4*nz];
		op[4][1] = opArray_im[idx+4*nz];
		*/


		// Construction and application of the nonlinear operator
		// Use the __fmul_rn() to avoid the combination into a FMAD by the compiler
		const float Nm_im = __fmul_rn(gamma,(powf(u[2][2],2.0)+powf(u[3][2],2.0) + __fmul_rn(2.0,(powf(u[0][2],2.0)+powf(u[1][2],2.0)))));
		const float Np_im = __fmul_rn(gamma,(powf(u[0][0],2.0)+powf(u[1][0],2.0) + __fmul_rn(2.0,(powf(u[2][0],2.0)+powf(u[3][0],2.0)))));
		const float um_nl_im = __fmul_rn(cosf(h*Nm_im),u[3][2]) + __fmul_rn(sinf(h*Nm_im),u[2][2]);
		const float um_nl_re = __fmul_rn(cosf(h*Nm_im),u[2][2]) - __fmul_rn(sinf(h*Nm_im),u[3][2]);
		const float up_nl_im = __fmul_rn(cosf(h*Np_im),u[1][0]) + __fmul_rn(sinf(h*Np_im),u[0][0]);
		const float up_nl_re = __fmul_rn(cosf(h*Np_im),u[0][0]) - __fmul_rn(sinf(h*Np_im),u[1][0]);

		/*
		// Application of the linear operator - Version global
		um_next_re = __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+1*nz],um_nl_re) - __fmul_rn(opArray_im[idx+1*nz],um_nl_im)));
		um_next_re += __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+2*nz],up_nl_re) - __fmul_rn(opArray_im[idx+2*nz],up_nl_im)));
		um_next_im = __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+1*nz],um_nl_im) + __fmul_rn(opArray_im[idx+1*nz],um_nl_re)));
		um_next_im += __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+2*nz],up_nl_im) + __fmul_rn(opArray_im[idx+2*nz],up_nl_re)));
		up_next_re = __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+3*nz],um_nl_re) - __fmul_rn(opArray_im[idx+3*nz],um_nl_im)));
		up_next_re += __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+4*nz],up_nl_re) - __fmul_rn(opArray_im[idx+4*nz],up_nl_im)));
		up_next_im = __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+3*nz],um_nl_im) + __fmul_rn(opArray_im[idx+3*nz],um_nl_re)));
		up_next_im += __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+4*nz],up_nl_im) + __fmul_rn(opArray_im[idx+4*nz],up_nl_re)));
		*/

		// Application of the linear operator - Version global
		um_next_re = __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+1*nz],um_nl_re) - __fmul_rn(opArray_im[idx+1*nz],um_nl_im)));
		um_next_re += __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+2*nz],up_nl_re) - __fmul_rn(opArray_im[idx+2*nz],up_nl_im)));
		um_next_im = __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+1*nz],um_nl_im) + __fmul_rn(opArray_im[idx+1*nz],um_nl_re)));
		um_next_im += __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+2*nz],up_nl_im) + __fmul_rn(opArray_im[idx+2*nz],up_nl_re)));
		up_next_re = __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+3*nz],um_nl_re) - __fmul_rn(opArray_im[idx+3*nz],um_nl_im)));
		up_next_re += __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+4*nz],up_nl_re) - __fmul_rn(opArray_im[idx+4*nz],up_nl_im)));
		up_next_im = __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+3*nz],um_nl_im) + __fmul_rn(opArray_im[idx+3*nz],um_nl_re)));
		up_next_im += __fmul_rn(opArray_re[idx+5*nz],(__fmul_rn(opArray_re[idx+4*nz],up_nl_im) + __fmul_rn(opArray_im[idx+4*nz],up_nl_re)));

		__syncthreads();

		up_re_o[idx] = up_next_re;
		up_im_o[idx] = up_next_im;
		um_re_o[idx] = um_next_re;
		um_im_o[idx] = um_next_im;
		
		//timeDetector[nt] = up_re[posDetector];
		//timeDetector_im[nt] = up_im[posDetector];
		
	  }

	""")

	####### Optimized SplitStep Method Kernel end #######
	#####################################################


	func = mod.get_function("splitstep")
	for i in arange(nt):

			"""
			paramatersInt[0] = nt
			paramatersInt_gpu = cuda.mem_alloc(paramatersInt.size*paramatersInt.dtype.itemsize)
			cuda.memcpy_htod(paramatersInt_gpu, paramatersInt)
			"""

			func(up_gpu, up_im_gpu, um_gpu, um_im_gpu,
				 up_gpu_o, up_im_gpu_o, um_gpu_o, um_im_gpu_o,
				 opArray_gpu, opArray_im_gpu,
				 timeDetector_gpu, timeDetector_im_gpu, block=dimBlock, grid=dimGrid)

			cuda.memcpy_dtod(up_gpu, up_gpu_o, up_float32.size*up_float32.dtype.itemsize)
			cuda.memcpy_dtod(up_im_gpu, up_im_gpu_o, up_float32.size*up_float32.dtype.itemsize)
			cuda.memcpy_dtod(um_gpu, um_gpu_o, up_float32.size*up_float32.dtype.itemsize)
			cuda.memcpy_dtod(um_im_gpu, um_im_gpu_o, up_float32.size*up_float32.dtype.itemsize)
	
			# Archive le signal avec une periode stepArchive
			if archiveFlag == stepArchive:
				cuda.memcpy_dtoh(upArchiveInt32[ia], up_gpu_o)
				cuda.memcpy_dtoh(umArchiveInt32[ia], um_gpu_o)
				cuda.memcpy_dtoh(up_im_ArchiveInt32[ia], up_im_gpu_o)
				cuda.memcpy_dtoh(um_im_ArchiveInt32[ia], um_im_gpu_o)
				upArchiveInt[ia] = pow(upArchiveInt32[ia],2) + pow(up_im_ArchiveInt32[ia],2)
				umArchiveInt[ia] = pow(umArchiveInt32[ia],2) + pow(um_im_ArchiveInt32[ia],2)
				ia += 1
				archiveFlag = 1
			else:
				archiveFlag += 1
			

 	# Recopy the result from the gpu memory to the main memory
	up_final32 = empty_like(up_float32)
	um_final32 = empty_like(um_float32)
	up_final32_im = empty_like(up_float32)
	um_final32_im = empty_like(um_float32)
	cuda.memcpy_dtoh(up_final32, up_gpu_o)
	cuda.memcpy_dtoh(um_final32, um_gpu_o)
	cuda.memcpy_dtoh(up_final32_im, up_im_gpu_o)
	cuda.memcpy_dtoh(um_final32_im, um_im_gpu_o)
	#cuda.memcpy_dtoh(timeDetector32 , timeDetector_gpu)
	#cuda.memcpy_dtoh(timeDetector32_im, timeDetector_im_gpu)

	##########################################

	# Convert to numpy complex type
	um_final = zeros(nz, complex)
	up_final = zeros(nz, complex)
	um_final.real = um_final32
	um_final.imag = um_final32_im
	up_final.real = up_final32
	up_final.imag = up_final32_im
	timeDetector.real = timeDetector32
	timeDetector.imag = timeDetector32_im

	return [up_final, um_final, upArchiveInt, umArchiveInt, timeDetector]


def ossfm(up, length, nt, alpha, beta1, kappa, detuning, gamma, stepArchive = 1, dz_over_dt = 1.0, posDectector = 0):

	'''	
	Implementation of the optimized split-step method for solving nonlinear 
	coupled-mode equations that model wave propagation in nonlinear fiber Bragg
	gratings. (version nonsymmetrized with fourier propagation)

	ref. Toroker et al. 'Optimized split-step method for modeling
	nonlinear pulse propagation in nonlinear fiber Bragg gratings'

	up: 		Profil spatial de l'impulsion
	length:		Longueur du fbg
	nt:			Nombre de steps temporel
	alpha:		Parametre de gain
	beta1:		Constante de propagation (1/vitesse de groupe)
	kappa:		Parametre de couplage du fbg [1/m]
	detuning:	Detuning par rapport a la longueur d'onde de bragg [1/m]
	'''

	# Vitesse de groupe dans le fibre sans fbg [m/ps]
	Vg = (1.0/beta1)/(1e12)
	nz = len(up)
	dz = length / nz
	k = wspace(dz*nz,nz)

	# Cas simple ou dt est fixee a dz par la vitesse de groupe [ps]
	dt = (dz / Vg) / dz_over_dt
	h = Vg * dt

	# On suppose que u_moins = 0 a z = L
	# Width of the absorbing layer (ratio of the total width)
	absLayer = 0.1
	um = zeros(nz, complex)
	upArchiveInt = zeros([nt/stepArchive,nz], double)
	umArchiveInt = zeros([nt/stepArchive,nz], double)
	timeDetector = zeros(nt, complex)

	# Construction of the windowing function
	absZone = int(absLayer*nz)
	W = zeros(nz, double)
	Wz = arange(1,nz+1)
	W[absZone:nz-absZone] = 1.0
	Wabs = pow(sin(pi*(Wz[0:absZone]+nz/2)/(2*absZone)),1.0/3)
	W[nz-absZone:nz] = Wabs
	W[0:absZone] = Wabs[::-1]


	# Construction de l'operateur lineaire
	Dmg = exp(h*(-alpha/2.0))
	Dm1 = (cos(detuning*h)+1.0j*sin(detuning*h))*cos(kappa*h)
	Dm2 = (cos(detuning*h)+1.0j*sin(detuning*h))*1.0j*sin(kappa*h)
	Dp1 = (cos(detuning*h)+1.0j*sin(detuning*h))*1.0j*sin(kappa*h)
	Dp2 = (cos(detuning*h)+1.0j*sin(detuning*h))*cos(kappa*h)
	
	archiveFlag = stepArchive
	ia = 0

	for it in arange(nt):
		# Application de l'operateur nonlineaire
		Nm = 1.0j*gamma*(pow(abs(shiftfourier(um, h, dz, nz)),2)+2*pow(abs(shiftfourier(up,h,dz,nz)),2))
		Np = 1.0j*gamma*(pow(abs(shiftfourier(up,-h, dz, nz)),2)+2*pow(abs(shiftfourier(um,-h, dz, nz)),2))
		um_nl = exp(h*Nm)*shiftfourier(um, h, dz, nz)
		up_nl = exp(h*Np)*shiftfourier(up,-h, dz, nz)

		# Application de l'operateur lineaire
		um_next = (Dmg * Dm1 * um_nl) + (Dmg * Dm2 * up_nl)
		up_next = (Dmg * Dp1 * um_nl) + (Dmg * Dp2 * up_nl)

		up = up_next*W
		um = um_next*W

		# Store the temporel profile with a "Scifi" amplitude detector
		timeDetector[it] = up[posDectector]

		# Archive le signal avec une periode stepArchive
		if archiveFlag == stepArchive:
			upArchiveInt[ia] = pow(abs(up),2)
			umArchiveInt[ia] = pow(abs(um),2)
			ia += 1
			archiveFlag = 1
		else:
			archiveFlag += 1

	return [up, um, upArchiveInt, umArchiveInt, timeDetector]


def ossmMovie(dataMatrix1, dataMatrix2, z, kappa, showFrame = 0, scaleAuto = 0):
	''' 
	Generate an animation with the ossm simulation results
	'''

	import matplotlib.collections as collections
	
	matrixSize1 = dataMatrix1.shape[1]
	matrixSize2 = dataMatrix2.shape[1]

	pl.figure(figsize=(10.6,6.0))
	ax = pl.subplot(111)

	line,line2 = pl.plot(z, dataMatrix1[:,1], z, dataMatrix2[:,1])

	maxValue = dataMatrix1.max()
	kappaMax = kappa.max()
	ax.set_ylim((0,maxValue))
	ax.set_xlim((z.min(),z.max()))
	ax.set_ylabel("Intensity [W]")
	ax.set_xlabel("FBG Lenght [mm]")
	
	collection = collections.BrokenBarHCollection.span_where(
		   z, ymin=0, ymax=maxValue, where=kappa>0, facecolor='blue', alpha=0.05)
	ax.add_collection(collection)
	
	if showFrame:
		pl.title("Movie - Frame: " + str(1))

	for i in pl.arange(matrixSize1):
	  	line.set_ydata(dataMatrix1[:,i])
	  	line2.set_ydata(dataMatrix2[:,i])
		if (scaleAuto==1):
			maxValue = dataMatrix1[:,i].max()
			ax.set_ylim((0,maxValue))
		if showFrame:
			pl.title("Movie - Frame: " + str(i))
    		pl.draw()


def ossmSumMovie(dataMatrix1, dataMatrix2, z, kappa, showFrame = 0, scaleAuto = 0):
	''' 
	Generate an animation with the ossm simulation results
	This version sum the two Bloch wave envelopes
	'''

	import matplotlib.collections as collections
	
	matrixSize1 = dataMatrix1.shape[1]
	matrixSize2 = dataMatrix2.shape[1]

	pl.figure(figsize=(10.6,6.0))
	ax = pl.subplot(111)

	line,line2 = pl.plot(z, dataMatrix1[:,1], z, dataMatrix2[:,1])

	maxValue = (dataMatrix1+dataMatrix2).max()
	kappaMax = kappa.max()
	ax.set_ylim((0,maxValue))
	ax.set_xlim((z.min(),z.max()))
	ax.set_ylabel("Intensity [W]")
	ax.set_xlabel("FBG Lenght [mm]")
	
	collection = collections.BrokenBarHCollection.span_where(
		   z, ymin=0, ymax=maxValue, where=kappa>0, facecolor='blue', alpha=0.05)
	ax.add_collection(collection)
	
	if showFrame:
		pl.title("Movie - Frame: " + str(1))

	for i in pl.arange(matrixSize1):
	  	line.set_ydata(dataMatrix1[:,i]+dataMatrix2[:,i])
		if (scaleAuto==1):
			maxValue = (dataMatrix1[:,i]+dataMatrix2[:,i]).max()
			ax.set_ylim((0,maxValue))
		if showFrame:
			pl.title("Movie - Frame: " + str(i))
    		pl.draw()


