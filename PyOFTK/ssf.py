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
import numexpr as ne
from scipy import *
from PyOFTK.utilities import *
import scipy.fftpack as fftpack
import ssprop

from pyfft.cuda import Plan
from pycuda.tools import make_default_context
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel

def vector(u0x, u0y, dt, dz, nz, alphaa, alphab, betaa, betap, gamma, psp, maxiter, tol, psi, chi, allStep = False):
	'''
	SSPROP symmetrized split-step fourier (vector)

	This function is a python wrapped around the sspropc code.
	More info here:

	http://www.photonics.umd.edu/software/ssprop
	'''
	nt = u0x.shape[0]
	if allStep:
		u0xdz = zeros([nz+1, nt], complex)
		u0xdz[0] = u0x
		u0ydz = zeros([nz+1, nt], complex)
		u0ydz[0] = u0y
		for z in (arange(nz)+1):
			[u0xdz[z], u0ydz[z], outputParam] = ssprop.vector(u0xdz[z-1], u0ydz[z-1], nt, nt, dt, dz, 1, alphaa, alphab, betap, betaa, gamma, psp, maxiter, tol, psi, chi, 0)
		u_x = u0xdz[nz]
		u_y = u0ydz[nz-1]

		if outputParam[9] == 0:
			return [u_x, u_y, outputParam, u0xdz[1:nz+1], u0ydz[1:nz+1]]
		else:
			raise Exception, "Failed to converge"

	else:
		[u_x, u_y, outputParam] = ssprop.vector(u0x, u0y, nt, nt, dt, dz, int(nz), alphaa, alphab, betap, betaa, gamma, psp, maxiter, tol, psi, chi, 0)
	
		if outputParam[9] == 0:
			return [u_x, u_y, outputParam]
		else:
			raise Exception, "Failed to converge"


def scalar(u0, dt, dz, nz, alpha, betap, gamma, tr = 0, to = 0, maxiter = 4, tol = 1e-5):
	'''
	SSPROP symmetrized split-step fourier (scalar).

	This function is a python wrapped around the sspropc code.
	More info here:

	http://www.photonics.umd.edu/software/ssprop/scalar.html
	'''
	nt = u0.shape[0]
	[u1, outputParam] = ssprop.scalar(u0, int(nt), dt, dz, int(nz), alpha, betap, gamma, tr, to, maxiter, tol, 0)
	if outputParam[9] == 0:
		return [u1, outputParam]
	else:
		raise Exception, "Failed to converge"


def ssf(u0, dt, dz, nz, alpha, betap, gamma, maxiter = 4, tol = 1e-5, phiNLOut = False):

	'''	
	Very simple implementation of the symmetrized split-step fourier algo.
	Solve the NLS equation with the SPM nonlinear terme only.

		* error: third in step size
		* u0 : Input field
		* dt: Time increment
		* dz: Space increment
		* nz: Number of space propagation step
		* alpha: Loss/Gain parameter
		* betap: Beta array beta[2] = GVD, beta[3] = TOD, etc...
		* gamma: Nonlinear parameter
		* maxiter: Maximal number of iteration per step (4)
		* tol: Error for each step (1e-5)
		* phiNLOut: If True return the nonlinear phase shift (True)

	'''	

	e_ini = pow(abs(u0),2).sum()
	nt = len(u0)
	w = wspace(dt*nt,nt)
	phiNL = 0.0

	# Construction de l'operateur lineaire
	halfstep = -alpha/2.0	
	if len(betap) != nt:
		for ii in arange(len(betap)):
			halfstep = halfstep - 1.0j*betap[ii]*pow(w,ii)/factorial(ii)
	halfstep = exp(halfstep*dz/2.0)

	u1 = u0

	ufft = fftpack.fft(u0)

	for iz in arange(nz):
		# First application of the linear operator
		uhalf = fftpack.ifft(halfstep*ufft)
		for ii in arange(maxiter):
			# Application de l'operateur nonlineaire en approx. l'integral de N(z)dz
			# avec la methode du trapeze
			uv = uhalf * exp(-1.0j*gamma*(pow(abs(u1),2.0) + pow(abs(u0),2.0))*dz/2.0)
			uv = fftpack.fft(uv)
			# Second application of the linear operator
			ufft = halfstep*uv
			uv = fftpack.ifft(ufft)

			error = linalg.norm(uv-u1,2.0)/linalg.norm(u1,2.0)

			if (error < tol ):
				u1 = uv
				break
			else:
				u1 = uv
		
		if (ii == maxiter):
			raise Exception, "Failed to converge"

		phiNL += gamma*dz*pow(abs(u1),2).max()
		
		u0 = u1

	if phiNLOut:
		return [u1, phiNL]
	else:
		return u1



def ssfu(up, dt, dz, nz, alpha, betap, gamma, maxiter = 4, tol = 1e-5):

	'''	
	Very simple implementation of the unsymmetrized split-step fourier algo.
	error: second in step size
	u0 : Input field
	'''	

	nt = len(up)
	w = wspace(dt*nt,nt)

	# Construction de l'operateur lineaire
	linearstep = -alpha/2.0	
	for ii in arange(len(betap)):
		linearstep = linearstep - 1.0j*betap[ii]*pow(w,ii)/factorial(ii)
	linearstep = exp(linearstep*dz)

	ufft = fftpack.fft(up)

	for iz in arange(nz):
		# 1er Application de l'operateur lineaire
		uhalf = fftpack.ifft(linearstep*ufft)

		# Application de l'operateur nonlineaire
		uv = uhalf * exp(-1.0j*gamma*(pow(abs(up),2.0))*dz)

		ufft = fftpack.fft(uv)
		up = uv


	return uv


def ssfgpu(u0, dt, dz, nz, alpha, betap, gamma, maxiter = 4, tol = 1e-5, phiNLOut = False):

	'''	
	Very simple implementation of the symmetrized split-step fourier algo.
	Solve the NLS equation with the SPM nonlinear terme only.

		* error: third in step size
		* u0 : Input field
		* dt: Time increment
		* dz: Space increment
		* nz: Number of space propagation step
		* alpha: Loss/Gain parameter (array)
		* betap: Beta array beta[2] = GVD, beta[3] = TOD, etc...
		* gamma: Nonlinear parameter
		* maxiter: Maximal number of iteration per step (4)
		* tol: Error for each step (1e-5)
		* phiNLOut: If True return the nonlinear phase shift (True)

		--- GPU Version (float precision) ---
	'''	

	nt = len(u0)
	e_ini = pow(abs(u0),2).sum()
	w = wspace(dt*nt,nt)
	phiNL = 0.0

	# Make sure u0 is in single precision
	u0=u0.astype(complex64)
	alpha=alpha.astype(complex64)
	u1 = u0
	uv = empty_like(u0)

	# Construction of the linear operator
	halfstep = -alpha/2.0	
	if len(betap) != nt:
		for ii in arange(len(betap)):
			halfstep = halfstep - 1.0j*betap[ii]*pow(w,ii)/factorial(ii)
	halfstep = exp(halfstep*dz/2.0).astype(complex64)

	# CUDA Kitchen sink
	cuda.init()
	context = make_default_context()
	fftPlan = Plan((1, nt), dtype=numpy.complex64)

	# Allocate memory to the device
	gpu_halfstep = gpuarray.to_gpu(halfstep)
	gpu_u0 = gpuarray.to_gpu(u0)
	gpu_u1 = gpuarray.to_gpu(u1)
	gpu_uhalf = gpuarray.empty_like(gpu_u0)
	gpu_uv = gpuarray.empty_like(gpu_u0)
	gpu_ufft = gpuarray.empty_like(gpu_u0)
		
	fftPlan.execute(gpu_u0, gpu_ufft)
	
	# GPU Kernel corresponding to the linear operator
	halfStepKernel = ElementwiseKernel("pycuda::complex<float> *u, pycuda::complex<float> *halfstep, pycuda::complex<float> *uhalf",
		"uhalf[i] = u[i] * halfstep[i]",
		"halfstep_linear",
		preamble="#include <pycuda-complex.hpp>",)
	
	# GPU Kernel corresponding to the nonlinear operator
	nlKernel = ElementwiseKernel("pycuda::complex<float> *uhalf, pycuda::complex<float> *u0, pycuda::complex<float> *u1, pycuda::complex<float> *uv, float gamma, float dz",
		"""
		float u0_int = pow(u0[i]._M_re,2) + pow(u0[i]._M_im,2);
		float u1_int = pow(u1[i]._M_re,2) + pow(u1[i]._M_im,2);
		float realArg = -gamma*(u1_int + u0_int)*dz;
		float euler1 = cos(realArg);
		float euler2 = sin(realArg);
		uv[i]._M_re = uhalf[i]._M_re * euler1 - uhalf[i]._M_im * euler2;
		uv[i]._M_im = uhalf[i]._M_im * euler1 + uhalf[i]._M_re * euler2;
		""",
		"halfstep_nonlinear",
		preamble="#include <pycuda-complex.hpp>",)
	
	# GPU reduction kernel computing the error between two complex array
	computeError = ReductionKernel(numpy.float32, neutral="0",
		reduce_expr="a+b", map_expr="pow(abs(a[i] - b[i]),2)",
		arguments="pycuda::complex<float> *a, pycuda::complex<float> *b",
		name="error_reduction",
		preamble="#include <pycuda-complex.hpp>",)

	# Perfom a deep copy of a complex gpuarray
	complexDeepCopy = ElementwiseKernel("pycuda::complex<float> *u1, pycuda::complex<float> *u2",
		"u1[i]._M_re = u2[i]._M_re;u1[i]._M_im = u2[i]._M_im",
		"gpuarray_deepcopy",
		preamble="#include <pycuda-complex.hpp>",)
	
	# Main Loop
	for iz in arange(nz):
		# First application of the linear operator
		halfStepKernel(gpu_ufft, gpu_halfstep, gpu_uhalf)
		fftPlan.execute(gpu_uhalf, inverse=True)
		for ii in arange(maxiter):
			# Application de l'operateur nonlineaire en approx. l'integral de N(z)dz
			# avec la methode du trapeze 
			nlKernel(gpu_uhalf, gpu_u0, gpu_u1, gpu_uv, float(gamma), float(dz/2.0))
			fftPlan.execute(gpu_uv)
			# Second application of the linear operator
			halfStepKernel(gpu_uv, gpu_halfstep, gpu_ufft)
			fftPlan.execute(gpu_ufft, gpu_uv, inverse=True)

			error = computeError(gpu_u1, gpu_uv).get() / e_ini

			if (error < tol):
				complexDeepCopy(gpu_u1, gpu_uv)
				break
			else:
				complexDeepCopy(gpu_u1, gpu_uv)
		
		if (ii >= maxiter-1):
			context.pop()
			raise Exception, "Failed to converge"

		complexDeepCopy(gpu_u0, gpu_u1)

	u1 = gpu_u1.get()
	context.pop()

	if phiNLOut:
		return [u1, phiNL]
	else:
		return u1


def ssfgpu_new(u0, dt, dz, nz, alpha, betap, gamma, maxiter = 4, tol = 1e-5, phiNLOut = False):

	'''	
	Very simple implementation of the symmetrized split-step fourier algo.
	Solve the NLS equation with the SPM nonlinear terme only.

		* error: third in step size
		* u0 : Input field
		* dt: Time increment
		* dz: Space increment
		* nz: Number of space propagation step
		* alpha: Loss/Gain parameter (array)
		* betap: Beta array beta[2] = GVD, beta[3] = TOD, etc...
		* gamma: Nonlinear parameter
		* maxiter: Maximal number of iteration per step (4)
		* tol: Error for each step (1e-5)
		* phiNLOut: If True return the nonlinear phase shift (True)

		--- GPU Version (float precision) ---
	'''	

	nt = len(u0)
	e_ini = pow(abs(u0),2).sum()
	w = wspace(dt*nt,nt)
	phiNL = 0.0

	# Make sure u0 is in single precision
	u0=u0.astype(complex64)
	alpha=alpha.astype(complex64)
	u1 = u0
	uv = empty_like(u0)

	# Construction of the linear operator
	halfstep = -alpha/2.0	
	if len(betap) != nt:
		for ii in arange(len(betap)):
			halfstep = halfstep - 1.0j*betap[ii]*pow(w,ii)/factorial(ii)
	halfstep = exp(halfstep*dz/2.0).astype(complex64)

	# CUDA Kitchen sink
	cuda.init()
	context = make_default_context()
	fftPlan = Plan((1, nt), dtype=numpy.complex64)

	# Allocate memory to the device
	gpu_halfstep = gpuarray.to_gpu(halfstep)
	gpu_u0 = gpuarray.to_gpu(u0)
	gpu_u1 = gpuarray.to_gpu(u1)
	gpu_uhalf = gpuarray.empty_like(gpu_u0)
	gpu_uv = gpuarray.empty_like(gpu_u0)
	gpu_ufft = gpuarray.empty_like(gpu_u0)
		
	fftPlan.execute(gpu_u0, gpu_ufft)
	
	# GPU Kernel corresponding to the linear operator
	halfStepKernel = ElementwiseKernel("pycuda::complex<float> *u, pycuda::complex<float> *halfstep, pycuda::complex<float> *uhalf",
		"uhalf[i] = u[i] * halfstep[i]",
		"halfstep_linear",
		preamble="#include <pycuda-complex.hpp>",)
	
	# GPU Kernel corresponding to the nonlinear operator
	nlKernel = ElementwiseKernel("pycuda::complex<float> *uhalf, pycuda::complex<float> *u0, pycuda::complex<float> *u1, pycuda::complex<float> *uv, float gamma, float dz",
		"""
		float u0_int = pow(u0[i]._M_re,2) + pow(u0[i]._M_im,2);
		float u1_int = pow(u1[i]._M_re,2) + pow(u1[i]._M_im,2);
		float realArg = -gamma*(u1_int + u0_int)*dz;
		float euler1 = cos(realArg);
		float euler2 = sin(realArg);
		uv[i]._M_re = uhalf[i]._M_re * euler1 - uhalf[i]._M_im * euler2;
		uv[i]._M_im = uhalf[i]._M_im * euler1 + uhalf[i]._M_re * euler2;
		""",
		"halfstep_nonlinear",
		preamble="#include <pycuda-complex.hpp>",)
	
	# GPU reduction kernel computing the error between two complex array
	computeError = ReductionKernel(numpy.float32, neutral="0",
		reduce_expr="a+b", map_expr="pow(abs(a[i] - b[i]),2)",
		arguments="pycuda::complex<float> *a, pycuda::complex<float> *b",
		name="error_reduction",
		preamble="#include <pycuda-complex.hpp>",)

	# Perfom a deep copy of a complex gpuarray
	complexDeepCopy = ElementwiseKernel("pycuda::complex<float> *u1, pycuda::complex<float> *u2",
		"u1[i]._M_re = u2[i]._M_re;u1[i]._M_im = u2[i]._M_im",
		"gpuarray_deepcopy",
		preamble="#include <pycuda-complex.hpp>",)
	
	# Main Loop
	for iz in arange(nz):
		ii  = 0
		# First application of the linear operator
		halfStepKernel(gpu_ufft, gpu_halfstep, gpu_uhalf)
		fftPlan.execute(gpu_uhalf, inverse=True)
		error = 1E5
		while (ii < maxiter) and (error > tol):
			# Application de l'operateur nonlineaire en approx. l'integral de N(z)dz
			# avec la methode du trapeze 
			nlKernel(gpu_uhalf, gpu_u0, gpu_u1, gpu_uv, float(gamma), float(dz/2.0))
			fftPlan.execute(gpu_uv)
			# Second application of the linear operator
			halfStepKernel(gpu_uv, gpu_halfstep, gpu_ufft)
			fftPlan.execute(gpu_ufft, gpu_uv, inverse=True)

			error = computeError(gpu_u1, gpu_uv).get() / e_ini
			complexDeepCopy(gpu_u1, gpu_uv)
			ii =+ 1
		
		if (error <= tol):
			context.pop()
			raise Exception, "Failed to converge"

		complexDeepCopy(gpu_u0, gpu_u1)

	u1 = gpu_u1.get()
	context.pop()

	if phiNLOut:
		return [u1, phiNL]
	else:
		return u1


def ssfgpu2(u0, dt, dz, nz, alpha, betap, gamma, context, maxiter = 4, tol = 1e-5, phiNLOut = False):

	'''	
	Very simple implementation of the symmetrized split-step fourier algo.
	Solve the NLS equation with the SPM nonlinear terme only.

		* error: third in step size
		* u0 : Input field
		* dt: Time increment
		* dz: Space increment
		* nz: Number of space propagation step
		* alpha: Loss/Gain parameter (array)
		* betap: Beta array beta[2] = GVD, beta[3] = TOD, etc...
		* gamma: Nonlinear parameter
		* maxiter: Maximal number of iteration per step (4)
		* tol: Error for each step (1e-5)
		* phiNLOut: If True return the nonlinear phase shift (True)

		--- GPU Version (float precision) ---
	'''	

	nt = len(u0)
	e_ini = pow(abs(u0),2).sum()
	w = wspace(dt*nt,nt)
	phiNL = 0.0

	# Make sure u0 is in single precision
	u0=u0.astype(complex64)
	alpha=alpha.astype(complex64)
	u1 = u0
	uv = empty_like(u0)

	# Construction of the linear operator
	halfstep = -alpha/2.0	
	if len(betap) != nt:
		for ii in arange(len(betap)):
			halfstep = halfstep - 1.0j*betap[ii]*pow(w,ii)/factorial(ii)
	halfstep = exp(halfstep*dz/2.0).astype(complex64)

	# CUDA Kitchen sink
	fftPlan = Plan((1, nt), dtype=numpy.complex64)

	# Allocate memory to the device
	gpu_halfstep = gpuarray.to_gpu(halfstep)
	gpu_u0 = gpuarray.to_gpu(u0)
	gpu_u1 = gpuarray.to_gpu(u1)
	gpu_uhalf = gpuarray.empty_like(gpu_u0)
	gpu_uv = gpuarray.empty_like(gpu_u0)
	gpu_ufft = gpuarray.empty_like(gpu_u0)
		
	fftPlan.execute(gpu_u0, gpu_ufft)
	
	# GPU Kernel corresponding to the linear operator
	halfStepKernel = ElementwiseKernel("pycuda::complex<float> *u, pycuda::complex<float> *halfstep, pycuda::complex<float> *uhalf",
		"uhalf[i] = u[i] * halfstep[i]",
		"halfstep_linear",
		preamble="#include <pycuda-complex.hpp>",)
	
	# GPU Kernel corresponding to the nonlinear operator
	nlKernel = ElementwiseKernel("pycuda::complex<float> *uhalf, pycuda::complex<float> *u0, pycuda::complex<float> *u1, pycuda::complex<float> *uv, float gamma, float dz",
		"""
		float u0_int = pow(u0[i]._M_re,2) + pow(u0[i]._M_im,2);
		float u1_int = pow(u1[i]._M_re,2) + pow(u1[i]._M_im,2);
		float realArg = -gamma*(u1_int + u0_int)*dz;
		float euler1 = cos(realArg);
		float euler2 = sin(realArg);
		uv[i]._M_re = uhalf[i]._M_re * euler1 - uhalf[i]._M_im * euler2;
		uv[i]._M_im = uhalf[i]._M_im * euler1 + uhalf[i]._M_re * euler2;
		""",
		"halfstep_nonlinear",
		preamble="#include <pycuda-complex.hpp>",)
	
	# GPU reduction kernel computing the error between two complex array
	computeError = ReductionKernel(numpy.float32, neutral="0",
		reduce_expr="a+b", map_expr="pow(abs(a[i] - b[i]),2)",
		arguments="pycuda::complex<float> *a, pycuda::complex<float> *b",
		name="error_reduction",
		preamble="#include <pycuda-complex.hpp>",)

	# Perfom a deep copy of a complex gpuarray
	complexDeepCopy = ElementwiseKernel("pycuda::complex<float> *u1, pycuda::complex<float> *u2",
		"u1[i]._M_re = u2[i]._M_re;u1[i]._M_im = u2[i]._M_im",
		"gpuarray_deepcopy",
		preamble="#include <pycuda-complex.hpp>",)
	
	# Main Loop
	for iz in arange(nz):
		# First application of the linear operator
		halfStepKernel(gpu_ufft, gpu_halfstep, gpu_uhalf)
		fftPlan.execute(gpu_uhalf, inverse=True)
		for ii in arange(maxiter):
			# Application de l'operateur nonlineaire en approx. l'integral de N(z)dz
			# avec la methode du trapeze 
			nlKernel(gpu_uhalf, gpu_u0, gpu_u1, gpu_uv, float(gamma), float(dz/2.0))
			fftPlan.execute(gpu_uv)
			# Second application of the linear operator
			halfStepKernel(gpu_uv, gpu_halfstep, gpu_ufft)
			fftPlan.execute(gpu_ufft, gpu_uv, inverse=True)

			error = computeError(gpu_u1, gpu_uv).get() / e_ini

			if (error < tol):
				complexDeepCopy(gpu_u1, gpu_uv)
				break
			else:
				complexDeepCopy(gpu_u1, gpu_uv)
		
		if (ii >= maxiter-1):
			raise Exception, "Failed to converge"

		complexDeepCopy(gpu_u0, gpu_u1)

	u1 = gpu_u1.get()

	if phiNLOut:
		return [u1, phiNL]
	else:
		return u1


def ssf_b(u0, dt, dz, nz, alpha, betap, gamma, maxiter = 4, tol = 1e-5, phiNLOut = False):

	'''	
	Very simple implementation of the symmetrized split-step fourier algo.
	Solve the NLS equation with the SPM nonlinear terme only.

		* error: third in step size
		* u0 : Input field
		* dt: Time increment
		* dz: Space increment
		* nz: Number of space propagation step
		* alpha: Loss/Gain parameter
		* betap: Beta array beta[2] = GVD, beta[3] = TOD, etc...
		* gamma: Nonlinear parameter
		* maxiter: Maximal number of iteration per step (4)
		* tol: Error for each step (1e-5)
		* phiNLOut: If True return the nonlinear phase shift (True)

	'''	

	e_ini = pow(abs(u0),2).sum()
	nt = len(u0)
	w = wspace(dt*nt,nt)
	phiNL = 0.0

	# Construction de l'operateur lineaire
	halfstep = -alpha/2.0	
	if len(betap) != nt:
		for ii in arange(len(betap)):
			halfstep = halfstep - 1.0j*betap[ii]*pow(w,ii)/factorial(ii)
	halfstep = exp(halfstep*dz/2.0)

	u1 = u0

	ufft = fftpack.fft(u0)

	for iz in arange(nz):
		# First application of the linear operator
		uhalf = fftpack.ifft(halfstep*ufft)
		for ii in arange(maxiter):
			# Application de l'operateur nonlineaire en approx. l'integral de N(z)dz
			# avec la methode du trapeze
			uv = uhalf * exp(-1.0j*gamma*(pow(abs(u1),2.0) + pow(abs(u0),2.0))*dz/2.0)
			uv = fftpack.fft(uv)
			# Second application of the linear operator
			ufft = halfstep*uv
			uv = fftpack.ifft(ufft)
			error = float(pow(linalg.norm(uv-u1,2.0),2.0)/ e_ini)
			#error = linalg.norm(uv-u1,2.0)/linalg.norm(u1,2.0)
			#print error
			if (error < tol ):
				u1 = uv
				break
			else:
				u1 = uv
		
		if (ii >= maxiter-1):
			raise Exception, "Failed to converge"

		phiNL += gamma*dz*pow(abs(u1),2).max()
		
		u0 = u1

	if phiNLOut:
		return [u1, phiNL]
	else:
		return u1



