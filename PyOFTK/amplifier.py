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
from scipy import integrate
from core import *


# Planck constant [J*s]
h = 6.626069E-34
# Speed of light in vacuum [m/s]
c = 2.998E8
# boltzmann constant [J/K]
k = 1.3807E-23
# Ambiant temp. in Kelvin
T = 294.2
# Yb-doped Glass transition energy
epsYb = (h*c)/(10000*100)


class Amplifier():
	'''
	Class representing a generic Rare-Earth doped fiber amplifier
		* fiber: Doped fiber
		* pumpWL: List of pump wavelength [wl1 forward, wl1 backward, wl2 forward ...]
		* pumpPower: List of pump power [pw1 forward, pw1 backward, pw2 forward ...]
		* signalWL: List of signal wavelength [wl1 forward, wl1 backward, wl2 forward ...]
		* signalPower: List of signal power [pw1 forward, pw1 backward, pw2 forward ...]
		* alpha: Background lost [not implemented yet...]
		* nbrSections: Number of longitudinal section
		* aseRes: Resolution of the ASE spectrum
	'''

	def __init__(self, fiber, pumpWL, pumpPower, signalWL, signalPower, nbrSections = 100, aseRes = 100):

		self.dopedFiber = fiber
		self.nbrSignal = len(signalWL)
		self.nbrPump = len(pumpWL)
		self.nbrAse = aseRes
		self.signalWL = signalWL
		self.pumpWL = pumpWL
		self.aseWL = linspace(self.dopedFiber.wlMin, self.dopedFiber.wlMax, aseRes)
		self.aseDeltaLambda = self.aseWL[1] - self.aseWL[0]
		self.delta_nu = abs(-c/(pow(self.aseWL*1E-6,2))*self.aseDeltaLambda*1e-6)

		[self.sigma_em_p, self.sigma_abs_p] = fiber.crossSection(pumpWL)
		[self.sigma_em_s, self.sigma_abs_s] = fiber.crossSection(signalWL)
		[self.sigma_em_ase, self.sigma_abs_ase] = fiber.crossSection(self.aseWL)

		self.alpha_s = fiber.bgLoss(signalWL)
		self.alpha_p = fiber.bgLoss(pumpWL)
		self.alpha_ase = fiber.bgLoss(self.aseWL)

		self.nbrSections = nbrSections
		self.z = linspace(0,fiber.length,nbrSections)
		self.dz = self.dopedFiber.length / nbrSections

		self.P_ase_f = zeros([self.nbrAse, nbrSections])
		self.P_ase_b = zeros([self.nbrAse, nbrSections])
		self.P_p_f = zeros([self.nbrPump, nbrSections])
		self.P_p_b = zeros([self.nbrPump, nbrSections])
		self.P_s_f = zeros([self.nbrSignal, nbrSections])
		self.P_s_b = zeros([self.nbrSignal, nbrSections])

		self.N2 = zeros(nbrSections)
		self.N1 = zeros(nbrSections)

		# Initiale Conditions
		self.initNoise = 1E-6
		self.P_p_f[:,0] = pumpPower[0:self.nbrPump]
		self.P_p_b[:,-1] = pumpPower[self.nbrPump:2*self.nbrPump]
		self.P_s_f[:,0] = signalPower[0:self.nbrSignal]
		self.P_s_b[:,-1] = signalPower[self.nbrSignal:2*self.nbrSignal]
		self.P_ase_f[:,0] = self.initNoise
		self.P_ase_b[:,-1] = self.initNoise
	
		self.error = 1.0
	

	def __repr__(self):
		return "Generic Rare-Earth doped fiber amplifier"

	

	def set_init_pumpPower(self, pumpPower):
		'''
		Set the initial condition for the pumpPower for each pump
		'''
		self.P_p_f[:,0] = pumpPower


	def get_pumpPower(self):
		'''
		Get the pump power
		'''
		return [self.P_p_f, self.P_p_b]


	def get_signalPower(self):
		'''
		Get the signal power
		'''
		return [self.P_s_f, self.P_s_b]


	def get_asePower(self):
		'''
		Get the ase power
		'''
		return [self.P_ase_f, self.P_ase_b]

	
	def get_aseSpectrum(self, units='linear'):
		'''
		Return the ASE spectrum in both direction
		'''

		[ase_f, ase_b]  = {
		  'linear': lambda: [self.P_ase_f[:,-1], self.P_ase_b[:,0]],
		  'dBm': lambda:[10*log10(self.P_ase_f[:,-1]), 10*log10(self.P_ase_b[:,0])],
		}[units]()

		return [ase_f, ase_b]


	def set_init_signalPower(self, signalPower):
		'''
		Set the initial condition for the pumpPower for each signal
		'''
		self.P_s_f[:,0] = signalPower


	def set_init_asePower(self, asePower):
		'''
		Set the initial condition for the pumpPower for each ASE signal
		'''
		self.P_ase_f[:,0] = asePower

	
	def invSptProfil(self):
		'''
		Compute the population inversion spatial profil
		'''

		N2 = zeros(self.nbrSections)
		N1 = zeros(self.nbrSections)
		pWL = self.pumpWL
		sWL = self.signalWL
		aseWL = self.aseWL


		# Construct the transition rate factor 1->3
		W13 = zeros(self.nbrSections)
		for m in arange(self.nbrPump):
			W13 += (self.sigma_abs_p[m] * (self.P_p_f[m,:]/(self.dopedFiber.width(pWL[m])*1E-12))) / (h*c/(pWL[m]*1E-6))
		for v in arange(self.nbrPump):
			W13 += (self.sigma_abs_p[v] * (self.P_p_b[v,:]/(self.dopedFiber.width(pWL[v])*1E-12))) / (h*c/(pWL[v]*1E-6))

		# Construct the transition rate factor 2->1
		W21 = 0.0
		for l in arange(self.nbrSignal):
			W21 += (self.sigma_em_s[l] * (self.P_s_f[l,:]/(self.dopedFiber.width(sWL[l])*1E-12))) / (h*c/(sWL[l]*1E-6))
		for u in arange(self.nbrSignal):
			W21 += (self.sigma_em_s[u] * (self.P_s_b[u,:]/(self.dopedFiber.width(sWL[u])*1E-12))) / (h*c/(sWL[u]*1E-6))

		for n in arange(self.nbrAse):
			W21 += (self.sigma_em_ase[n] * (self.P_ase_f[n,:]/(self.dopedFiber.width(aseWL[n])*1E-12))) / (h*c/(aseWL[n]*1E-6))
		for v in arange(self.nbrAse):
			W21 += (self.sigma_em_ase[v] * (self.P_ase_b[v,:]/(self.dopedFiber.width(aseWL[v])*1E-12))) / (h*c/(aseWL[v]*1E-6))

		# Construct the transition rate factor 1->2
		W12 = 0.0
		for l in arange(self.nbrSignal):
			W12 += (self.sigma_abs_s[l] * (self.P_s_f[l,:]/(self.dopedFiber.width(sWL[l])*1E-12))) / (h*c/(sWL[l]*1E-6))
		for u in arange(self.nbrSignal):
			W12 += (self.sigma_abs_s[u] * (self.P_s_b[u,:]/(self.dopedFiber.width(sWL[u])*1E-12))) / (h*c/(sWL[u]*1E-6))

		for n in arange(self.nbrAse):
			W12 += (self.sigma_abs_ase[n] * (self.P_ase_f[n,:]/(self.dopedFiber.width(aseWL[n])*1E-12))) / (h*c/(aseWL[n]*1E-6))
		for v in arange(self.nbrAse):
			W12 += (self.sigma_abs_ase[v] * (self.P_ase_b[v,:]/(self.dopedFiber.width(aseWL[v])*1E-12))) / (h*c/(aseWL[v]*1E-6))

		# Compute the level population
		N2 = self.dopedFiber.concDopant * ( (W13 + W12) / ((1/self.dopedFiber.tau) + W21 + W12 + W13) )
		N1 = self.dopedFiber.concDopant - N2
		
		self.N1 = N1
		self.N2 = N2


	def simulate(self, direction=1, backwardOutput=False):

		def dPdz(w, z, sigma_abs_p, sigma_em_s, sigma_abs_s,
						 sigma_abs_ase, sigma_em_ase, Fiber, pWL, sWL, aseWL,
						 alpha_s, alpha_p, alpha_ase, delta_nu):
			'''
			RHS of the ODE systems
			'''				
			
			P_s_f = w[0:self.nbrSignal]
			P_s_b = w[self.nbrSignal:2*self.nbrSignal]

			P_p_f = w[2*self.nbrSignal:2*self.nbrSignal+self.nbrPump]
			P_p_b = w[2*self.nbrSignal+self.nbrPump:2*self.nbrSignal+2*self.nbrPump]
	
			P_ase_f = w[2*self.nbrSignal+2*self.nbrPump:2*self.nbrSignal+2*self.nbrPump+self.nbrAse]
			P_ase_b = w[2*self.nbrSignal+2*self.nbrPump+self.nbrAse:2*self.nbrSignal+2*self.nbrPump+2*self.nbrAse]

			# Construct the transition rate factor 1->3
			W13 = 0.0
			for m in arange(self.nbrPump):
				W13 += (sigma_abs_p[m] * (P_p_f[m]/(Fiber.width(pWL[m])*1E-12))) / (h*c/(pWL[m]*1E-6))
			for v in arange(self.nbrPump):
				W13 += (sigma_abs_p[v] * (P_p_b[v]/(Fiber.width(pWL[v])*1E-12))) / (h*c/(pWL[v]*1E-6))

			# Construct the transition rate factor 2->1
			W21 = 0.0
			for l in arange(self.nbrSignal):
				W21 += (sigma_em_s[l] * (P_s_f[l]/(Fiber.width(sWL[l])*1E-12))) / (h*c/(sWL[l]*1E-6))
			for u in arange(self.nbrSignal):
				W21 += (sigma_em_s[u] * (P_s_b[u]/(Fiber.width(sWL[u])*1E-12))) / (h*c/(sWL[u]*1E-6))

			for n in arange(self.nbrAse):
				W21 += (sigma_em_ase[n] * (P_ase_f[n]/(Fiber.width(aseWL[n])*1E-12))) / (h*c/(aseWL[n]*1E-6))
			for v in arange(self.nbrAse):
				W21 += (sigma_em_ase[v] * (P_ase_b[v]/(Fiber.width(aseWL[v])*1E-12))) / (h*c/(aseWL[v]*1E-6))

			# Construct the transition rate factor 1->2
			W12 = 0.0
			for l in arange(self.nbrSignal):
				W12 += (sigma_abs_s[l] * (P_s_f[l]/(Fiber.width(sWL[l])*1E-12))) / (h*c/(sWL[l]*1E-6))
			for u in arange(self.nbrSignal):
				W12 += (sigma_abs_s[u] * (P_s_b[u]/(Fiber.width(sWL[u])*1E-12))) / (h*c/(sWL[u]*1E-6))

			for n in arange(self.nbrAse):
				W12 += (sigma_abs_ase[n] * (P_ase_f[n]/(Fiber.width(aseWL[n])*1E-12))) / (h*c/(aseWL[n]*1E-6))
			for v in arange(self.nbrAse):
				W12 += (sigma_abs_ase[v] * (P_ase_b[v]/(Fiber.width(aseWL[v])*1E-12))) / (h*c/(aseWL[v]*1E-6))

			# Compute the level population
			N2 = Fiber.concDopant * ( (W13 + W12) / ((1/Fiber.tau) + W21 + W12 + W13) )
			N1 = Fiber.concDopant - N2

			P = zeros(2*self.nbrSignal+2*self.nbrPump+2*self.nbrAse)
			i = 0

			# Signal Power
			for l in arange(self.nbrSignal):
				P[i] = sign(direction)*(sigma_em_s[l]*N2 - sigma_abs_s[l]*N1 - alpha_s) * P_s_f[l] * Fiber.modeOverlap(sWL[l])
				i += 1
			for u in arange(self.nbrSignal):
				P[i] = -sign(direction)*(sigma_em_s[u]*N2 - sigma_abs_s[u]*N1 - alpha_s) * P_s_b[u] * Fiber.modeOverlap(sWL[u])
				i += 1

			# Pump Power
			for m in arange(self.nbrPump):
				P[i] = sign(direction)*(-sigma_abs_p[m]*N1 - alpha_p) * P_p_f[m] * Fiber.modeOverlap(pWL[m])
				i += 1
			for v in arange(self.nbrPump):
				P[i] = -sign(direction)*(-sigma_abs_p[v]*N1 - alpha_p) * P_p_b[v] * Fiber.modeOverlap(pWL[v])
				i += 1

			# ASE Power
			for n in arange(self.nbrAse):
				P[i] = sign(direction)*(sigma_em_ase[n]*N2 - sigma_abs_ase[n]*N1 - alpha_ase) * P_ase_f[n] * Fiber.modeOverlap(aseWL[n])
				P[i] += sign(direction)*2*(h*c/(aseWL[n]*1E-6)) * delta_nu[n] * sigma_em_ase[n]*N2 * Fiber.modeOverlap(aseWL[n])
				i += 1

			for v in arange(self.nbrAse):
				P[i] = -sign(direction)*(sigma_em_ase[v]*N2 - sigma_abs_ase[v]*N1 - alpha_ase) * P_ase_b[v] * Fiber.modeOverlap(aseWL[v])
				i += 1

			return P


		def chi2(array1, array2):
			'''
			Evaluate the error between two arrays with the chi2
			'''
			nbrPoints = shape(array1)[0]
			return sqrt( pow((array2-array1),2).sum() ) / nbrPoints


		arguments = (self.sigma_abs_p, self.sigma_em_s, self.sigma_abs_s,
					self.sigma_abs_ase, self.sigma_em_ase, self.dopedFiber,
					self.pumpWL, self.signalWL, self.aseWL,
					self.alpha_s, self.alpha_p, self.alpha_ase, self.delta_nu)

		# Set the initials conditions and resolve the ode system
		if sign(direction) == 1:
			w0 = r_[self.P_s_f[:,0],self.P_s_b[:,0],self.P_p_f[:,0],self.P_p_b[:,0],self.P_ase_f[:,0],self.P_ase_b[:,0]]
		else:
			w0 = r_[self.P_s_f[:,-1],self.P_s_b[:,-1],self.P_p_f[:,-1],self.P_p_b[:,-1],self.P_ase_f[:,-1],self.P_ase_b[:,-1]]

		solution = integrate.odeint(dPdz, w0, self.z, args=arguments)	

		self.P_s_f = solution[:,0:self.nbrSignal].T
		self.P_p_f = solution[:,2*self.nbrSignal:2*self.nbrSignal+self.nbrPump].T
		self.P_ase_f = solution[:,2*self.nbrSignal+2*self.nbrPump:2*self.nbrSignal+2*self.nbrPump+self.nbrAse].T

		if backwardOutput:		
			self.P_p_b = solution[:,2*self.nbrSignal+self.nbrPump:2*self.nbrSignal+2*self.nbrPump].T
			self.P_s_b = solution[:,self.nbrSignal:2*self.nbrSignal].T
			self.P_ase_b = solution[:,2*self.nbrSignal+2*self.nbrPump+self.nbrAse:2*self.nbrSignal+2*self.nbrPump+2*self.nbrAse].T		
		
		# Use the chi2 between the backward ASE signals computed in two different iterations to evaluate the convergence
		ase_b = solution[:,2*self.nbrSignal+2*self.nbrPump+self.nbrAse:2*self.nbrSignal+2*self.nbrPump+2*self.nbrAse].T[:,-1]
		self.error = chi2(self.P_p_b[:,-1], ase_b)


	def simulateBackward(self, direction=1):
		'''
		Propagate the signal in backward direction using the population
		found in the previous forward iteration. Since N2 and N1 are constant
		we can solve each equations with a simple integration
		'''		

		# Get the initiale conditions
		Pp_ini = self.P_p_b[:,-1]
		Ps_ini = self.P_s_b[:,-1]
		Pase_ini = self.P_ase_b[:,-1]
		
		self.invSptProfil()

		for m in arange(self.nbrPump):
			integrant = sign(direction)*(-self.sigma_abs_p[m]*self.N1[::-1] - self.alpha_p) * self.dopedFiber.modeOverlap(self.pumpWL[m])
			self.P_p_b[m,::-1] = r_[Pp_ini[m], Pp_ini[m]*exp(integrate.cumtrapz(integrant, self.z))]

		for l in arange(self.nbrSignal):
			integrant = sign(direction)*(self.sigma_em_s[l]*self.N2[::-1] - self.sigma_abs_s[l]*self.N1[::-1] - self.alpha_s)
			integrant *= self.dopedFiber.modeOverlap(self.signalWL[l])
			self.P_s_b[l,::-1] = r_[Ps_ini[l], Ps_ini[l]*exp(integrate.cumtrapz(integrant, self.z))]

		for v in arange(self.nbrAse):
			integrant = sign(direction)*(self.sigma_em_ase[v]*self.N2[::-1] - self.sigma_abs_ase[v]*self.N1[::-1] - self.alpha_ase)
			integrant *= self.dopedFiber.modeOverlap(self.aseWL[v])
			integrant2 = sign(direction)*2*(h*c/(self.aseWL[v]*1E-6)) * self.delta_nu[v] * self.sigma_em_ase[v]*self.N2[::-1]
			integrant2 *= self.dopedFiber.modeOverlap(self.aseWL[v])

			sol = integrate.cumtrapz(integrant, self.z)
			solTerme1 = exp(sol)
			solTerme1b = r_[1.0, exp(-sol)]
			solTerme2 = solTerme1 * integrate.cumtrapz(integrant2*solTerme1b, self.z)
			self.P_ase_b[v,::-1] = r_[Pase_ini[v], Pase_ini[v]*solTerme1 + solTerme2]

	
	def run(self, errorTol, nbrItrMax, errorOutput = False, verbose=False):
		
		error = zeros(nbrItrMax)

		self.simulate()
		for i in arange(nbrItrMax):
			self.simulateBackward()
			self.simulate()
			error[i] = self.error
			if verbose:
				print error[i]

		return error


class erbiumAmplifierSimple():
	'''
	Class representing a very simple erbium amplifier 1 pump, 1 signal forward
	'''

	def __init__(self, fiber, pumpPower, signalPower, nbrSections = 100):
		# Physical properties of the amplifier
		self.dopedFiber = fiber
		self.signalWL = 1.55
		self.pumpWL = 0.980
		self.aseWL = 1.50
		self.sigma_abs_p = 2.7E-25
		self.sigma_em_p = 0.0
		self.sigma_em_s = 2.52E-25
		self.sigma_abs_s = 1.98E-25
		[self.sigma_em_ase, self.sigma_abs_ase] = fiber.crossSection(self.aseWL)
		self.alpha_s = 0.0
		self.alpha_p = 0.0
		self.alpha_ase = 0.0
		self.delta_nu = 125E-9

		self.nbrSections = nbrSections
		self.z = linspace(0,fiber.length,nbrSections)
		self.dz = self.dopedFiber.length / nbrSections
		self.P_p_in = zeros(nbrSections)
		self.P_s_in = zeros(nbrSections)
		self.P_ase_in = zeros(nbrSections)
		self.P_p_out = zeros(nbrSections)
		self.P_s_out = zeros(nbrSections)
		self.P_ase_out = zeros(nbrSections)

		# Initiale Conditions
		self.P_p_out[0] = pumpPower
		self.P_s_out[0] = signalPower
		self.P_ase_out[0] = 0.0
	

	def __repr__(self):
		return "Very simple erbium-doped fiber amplifier"

	def set_init_pumpPower(self, pumpPower):
		self.P_p_out[0] = pumpPower

	def set_init_signalPower(self, signalPower):
		self.P_s_out[0] = signalPower

	def set_init_asePower(self, asePower):
		self.P_ase_out[0] = asePower

	def inversion(self, P_p, P_s, P_ase):
		'''
		Compute the population inversion with input Power
		'''
		W13 = (self.sigma_abs_p * (P_p/(self.dopedFiber.width(self.pumpWL)*1E-12))) / (h*c/(self.pumpWL*1E-6))
		W21 = (self.sigma_em_s * (P_s/(self.dopedFiber.width(self.signalWL)*1E-12))) / (h*c/(self.signalWL*1E-6))
		W21 += (self.sigma_em_ase * (P_ase/(self.dopedFiber.width(self.aseWL)*1E-12))) / (h*c/(self.aseWL*1E-6))
		W12 = (self.sigma_abs_s * (P_s/(self.dopedFiber.width(self.signalWL)*1E-12))) / (h*c/(self.signalWL*1E-6))
		W12 += (self.sigma_abs_ase * (P_ase/(self.dopedFiber.width(self.aseWL)*1E-12))) / (h*c/(self.aseWL*1E-6))

		N2 = self.dopedFiber.concDopant * ( (W13 + W12) / ((1/self.dopedFiber.tau) + W21 + W12 + W13) )
		N1 = self.dopedFiber.concDopant - N2

		return [N1,N2]


	def computeSection(self, s):

		[N1,N2] = self.inversion(self.P_p_out[s-1], self.P_s_out[s-1], self.P_ase_out[s-1])
		w0 = array([self.P_p_out[s-1], self.P_s_out[s-1], self.P_ase_out[s-1]])
		self.P_s_out[s] = self.P_s_out[s-1] + (self.sigma_em_s*N2 - self.sigma_abs_s*N1 - self.alpha_s) * self.P_s_out[s-1] * self.dopedFiber.modeOverlap(self.signalWL) * self.dz
		self.P_p_out[s] = self.P_p_out[s-1] + (-self.sigma_abs_p*N1 - self.alpha_p) * self.P_p_out[s-1] * self.dopedFiber.modeOverlap(self.pumpWL) * self.dz
		self.P_ase_out[s] = self.P_ase_out[s-1] + (self.sigma_em_ase*N2 - self.sigma_abs_ase*N1 - self.alpha_ase) * self.P_ase_out[s-1] * self.dopedFiber.modeOverlap(self.aseWL) * self.dz
		self.P_ase_out[s] += 2*(h*c/(self.aseWL*1E-6)) * self.delta_nu * self.sigma_em_ase*N2 * self.dopedFiber.modeOverlap(self.aseWL) * self.dz

	
	def simulateBySections(self):
		'''
		Compute all section
		'''
		for i in arange(1,self.nbrSections):
			self.computeSection(i)


	def simulate(self):

			def diffEquations(w, z, sigma_abs_p, sigma_em_s, sigma_abs_s,
							 sigma_abs_ase, sigma_em_ase, Fiber, pWL, sWL, aseWL,
							 alpha_s, alpha_p, alpha_ase, delta_nu):

				P_s, P_p, P_ase = w

				W13 = (sigma_abs_p * (P_p/(Fiber.width(pWL)*1E-12))) / (h*c/(pWL*1E-6))
				W21 = (sigma_em_s * (P_s/(Fiber.width(sWL)*1E-12))) / (h*c/(sWL*1E-6))
				W21 += (sigma_em_ase * (P_ase/(Fiber.width(aseWL)*1E-12))) / (h*c/(aseWL*1E-6))
				W12 = (sigma_abs_s * (P_s/(Fiber.width(sWL)*1E-12))) / (h*c/(sWL*1E-6))
				W12 += (sigma_abs_ase * (P_ase/(Fiber.width(aseWL)*1E-12))) / (h*c/(aseWL*1E-6))

				N2 = Fiber.concDopant * ( (W13 + W12) / ((1/Fiber.tau) + W21 + W12 + W13) )
				N1 = Fiber.concDopant - N2

				Ps = (sigma_em_s*N2 - sigma_abs_s*N1 - alpha_s) * P_s * Fiber.modeOverlap(sWL)
				Pp = (-sigma_abs_p*N1 - alpha_p) * P_p * Fiber.modeOverlap(pWL)
				Pase = (sigma_em_ase*N2 - sigma_abs_ase*N1 - alpha_ase) * P_ase * Fiber.modeOverlap(aseWL)
				Pase += 2*(h*c/(aseWL*1E-6)) * delta_nu * sigma_em_ase*N2 * Fiber.modeOverlap(aseWL)

				return [Ps, Pp, Pase]

			w0 = array([self.P_s_out[0], self.P_p_out[0], self.P_ase_out[0]])

			arguments = (self.sigma_abs_p, self.sigma_em_s, self.sigma_abs_s,
						self.sigma_abs_ase, self.sigma_em_ase, self.dopedFiber,
						self.pumpWL, self.signalWL, self.aseWL,
						self.alpha_s, self.alpha_p, self.alpha_ase, self.delta_nu)

			solution = integrate.odeint(diffEquations, w0, self.z, args=arguments)

			self.P_s_out = solution[:,0]
			self.P_p_out = solution[:,1]
			self.P_ase_out = solution[:,2]



