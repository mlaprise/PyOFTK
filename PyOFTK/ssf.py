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
from PyOFTK.utilities import *
import scipy.fftpack as fftpack
import ssprop


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

			if (linalg.norm(uv-u1,2.0)/linalg.norm(u1,2.0) < tol ):
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
