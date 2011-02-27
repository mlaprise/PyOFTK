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
from scipy import stats
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


class linearCavity():
	'''
	Class representing a generic linear Rare-Earth doped fiber laser
		* fiber: Doped fiber
		* pumpWL: List of pump wavelength [wl1 forward, wl1 backward, wl2 forward ...]
		* pumpPower: List of pump power [pw1 forward, pw1 backward, pw2 forward ...]
		* nbrSections: Number of longitudinal section
		* aseRes: Resolution of the ASE spectrum
	'''

	def __init__(self, fiber, pumpWL, pumpPower, fbg1, fbg2, nbrSections = 100, aseRes = 100):


		self.dopedFiber = fiber
	
		if isinstance(fiber, YbDopedFiber or YbDopedDCOF) or (isinstance(fiber, ErDopedFiber) and pumpWL > 1.450):	
			self.nbrSignal = len(pumpWL)
			self.nbrPump = 0
			self.pumpWL = 0.980
			self.signalWL = pumpWL
		elif isinstance(fiber, ErDopedFiber):
			self.nbrSignal = 0
			self.nbrPump = len(pumpWL)
			self.pumpWL = pumpWL
			self.signalWL = 1.55
		else:
			raise TypeError

		self.nbrAse = aseRes
		self.aseWL = linspace(self.dopedFiber.wlMin, self.dopedFiber.wlMax, aseRes)
		self.aseDeltaLambda = self.aseWL[1] - self.aseWL[0]
		self.delta_nu = abs(-c/(pow(self.aseWL*1E-6,2))*self.aseDeltaLambda*1e-6)
		self.fbg1 = fbg1
		self.fbg2 = fbg2

		#self.R1 = fbg1.reflectivity(self.aseWL)
		#self.R2 = fbg2.reflectivity(self.aseWL)
		
		# Construct the reflectivity curve with the different FBGs
		self.R1 = zeros(aseRes) + 0.04
		if isinstance(fbg1, list):
			for fbg in fbg1:
				wlDiff1 = abs(self.aseWL-fbg.braggWavelength)
				self.peakR1 = where(wlDiff1==wlDiff1.min())
				self.R1[self.peakR1] = fbg.maxReflectivity()
		else:
			wlDiff1 = abs(self.aseWL-fbg1.braggWavelength)
			self.peakR1 = where(wlDiff1==wlDiff1.min())
			self.R1[self.peakR1] = fbg1.maxReflectivity()
	
		self.R2 = zeros(aseRes) + 0.04
		if isinstance(fbg2, list):
			self.peakR2 = self.peakR1
			for fbg in fbg2:
				wlDiff2 = abs(self.aseWL-fbg.braggWavelength)
				self.peakR2 = where(wlDiff2==wlDiff2.min())
				self.R2[self.peakR2] = fbg.maxReflectivity()
		else:
			wlDiff2 = abs(self.aseWL-fbg2.braggWavelength)
			self.peakR2 = where(wlDiff2==wlDiff2.min())
			self.R2[self.peakR2] = fbg2.maxReflectivity()


		[self.sigma_em_p, self.sigma_abs_p] = fiber.crossSection(self.pumpWL)
		[self.sigma_em_s, self.sigma_abs_s] = fiber.crossSection(self.signalWL)
		[self.sigma_em_ase, self.sigma_abs_ase] = fiber.crossSection(self.aseWL)

		self.alpha_s = fiber.bgLoss(self.signalWL)
		self.alpha_p = fiber.bgLoss(self.pumpWL)
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
		self.output = zeros(self.nbrAse)

		self.N2 = zeros(nbrSections)
		self.N1 = zeros(nbrSections)

		# Initiale Conditions
		self.P_ase_f[:,0] = 0.0
		self.P_ase_b[:,-1] = 0.0

		if isinstance(fiber, YbDopedFiber or YbDopedDCOF) or (isinstance(fiber, ErDopedFiber) and pumpWL > 1.450):
			self.P_s_f[:,0] = pumpPower[0:self.nbrSignal]
			self.P_s_b[:,-1] =  pumpPower[self.nbrSignal:2*self.nbrSignal]
		elif isinstance(fiber, ErDopedFiber):
			self.P_p_f[:,0] = pumpPower[0:self.nbrPump]
			self.P_p_b[:,-1] = pumpPower[self.nbrPump:2*self.nbrPump]
		else:
			raise TypeError
	
		self.error = 1.0
	

	def info(self):
		print


	def __repr__(self):
		return "Generic linear Rare-Earth doped fiber laser"


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

	
	def get_outputPower(self,units='linear'):
	
		integral = integrate.simps(self.output)
		outputPower  = {
		  'linear': lambda: integral,
		  'dBm': lambda:10*log10(integral),
		}[units]()

		return outputPower


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


	
	def get_outputSpectrum(self, units='linear'):
		'''
		Integrate the ASE signal an return the spectrum in both direction
		'''
		[ase_f, ase_b]  = {
		  'linear': lambda: [self.P_ase_f[:,-1]*(1-self.R2), self.P_ase_b[:,0]*(1-self.R1)],
		  'dBm': lambda:[10*log10(self.P_ase_f[:,-1]*(1-self.R2)), 10*log10(self.P_ase_b[:,0]*(1-self.R1))],
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


	def set_wl_range(self, minWL, maxWL):
		'''
		Set the wavelength range of the simulation
		'''
		self.aseWL = linspace(minWL, maxWL, self.nbrAse)
		self.aseDeltaLambda = self.aseWL[1] - self.aseWL[0]
		self.R1 = self.fbg1.reflectivity(self.aseWL)
		self.R2 = self.fbg2.reflectivity(self.aseWL)


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

		return [N2, N1]


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
				P[i] = sign(direction)*(sigma_em_s[l]*N2 - sigma_abs_s[l]*N1 - alpha_s) * P_s_f[l] * Fiber.pumpOverlap(sWL[l])
				i += 1
			for u in arange(self.nbrSignal):
				P[i] = -sign(direction)*(sigma_em_s[u]*N2 - sigma_abs_s[u]*N1 - alpha_s) * P_s_b[u] * Fiber.pumpOverlap(sWL[u])
				i += 1

			# Pump Power
			for m in arange(self.nbrPump):
				P[i] = sign(direction)*(-sigma_abs_p[m]*N1 - alpha_p) * P_p_f[m] * Fiber.pumpOverlap(pWL[m])
				i += 1
			for v in arange(self.nbrPump):
				P[i] = -sign(direction)*(-sigma_abs_p[v]*N1 - alpha_p) * P_p_b[v] * Fiber.pumpOverlap(pWL[v])
				i += 1

			# ASE Power
			for n in arange(self.nbrAse):
				P[i] = sign(direction)*(sigma_em_ase[n]*N2 - sigma_abs_ase[n]*N1 - alpha_ase) * P_ase_f[n] * Fiber.modeOverlap(aseWL[n])
				P[i] += sign(direction)*2*(h*c/(aseWL[n]*1E-6)) * delta_nu[n] * sigma_em_ase[n]*N2 * Fiber.modeOverlap(aseWL[n])
				i += 1

			for v in arange(self.nbrAse):
				P[i] = -sign(direction)*(sigma_em_ase[v]*N2 - sigma_abs_ase[v]*N1 - alpha_ase) * P_ase_b[v] * Fiber.modeOverlap(aseWL[v])
				P[i] += -sign(direction)*2*(h*c/(aseWL[v]*1E-6)) * delta_nu[v] * sigma_em_ase[v]*N2 * Fiber.modeOverlap(aseWL[v])
				i += 1

			return P


		arguments = (self.sigma_abs_p, self.sigma_em_s, self.sigma_abs_s,
					self.sigma_abs_ase, self.sigma_em_ase, self.dopedFiber,
					self.pumpWL, self.signalWL, self.aseWL,
					self.alpha_s, self.alpha_p, self.alpha_ase, self.delta_nu)

		# Set the initials conditions and resolve the ode system
		if sign(direction) == 1:
			w0 = r_[self.P_s_f[:,0],self.P_s_b[:,0],self.P_p_f[:,0],self.P_p_b[:,0],self.P_ase_b[:,0]*self.R1,self.P_ase_b[:,0]]
		else:
			w0 = r_[self.P_s_f[:,-1],self.P_s_b[:,-1],self.P_p_f[:,-1],self.P_p_b[:,-1],self.P_ase_f[:,-1],self.P_ase_b[:,-1]]

		solution = integrate.odeint(dPdz, w0, self.z, args=arguments)	

		self.P_s_f = solution[:,0:self.nbrSignal].T
		self.P_p_f = solution[:,2*self.nbrSignal:2*self.nbrSignal+self.nbrPump].T
		self.P_ase_f = solution[:,2*self.nbrSignal+2*self.nbrPump:2*self.nbrSignal+2*self.nbrPump+self.nbrAse].T



	def simulateBackward(self, direction=1):
		'''
		Propagate the signal in backward direction using the population
		found in the previous forward iteration. Since N2 and N1 are constant
		we can solve each equations with a simple integration
		'''	

		# Get the initiale conditions
		Pp_ini = self.P_p_b[:,-1]
		Ps_ini = self.P_s_b[:,-1]
		Pase_ini = self.P_ase_f[:,-1]*self.R2
		
		self.invSptProfil()

		for m in arange(self.nbrPump):
			integrant = sign(direction)*(-self.sigma_abs_p[m]*self.N1[::-1] - self.alpha_p) * self.dopedFiber.pumpOverlap(self.pumpWL[m])
			self.P_p_b[m,::-1] = r_[Pp_ini[m], Pp_ini[m]*exp(integrate.cumtrapz(integrant, self.z))]

		for l in arange(self.nbrSignal):
			integrant = sign(direction)*(self.sigma_em_s[l]*self.N2[::-1] - self.sigma_abs_s[l]*self.N1[::-1] - self.alpha_s)
			integrant *= self.dopedFiber.pumpOverlap(self.signalWL[l])
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


	def run(self, errorTol, nbrItrMax, avgVar = 5, errorOutput = False, verbose = False):
		'''
		Compute the laser output by solving the ode systems with a relaxation method
		'''		

		i = 0
		normVar = 1.0 + errorTol

		if nbrItrMax < avgVar:
			nbrItrMax = avgVar

		outputError = zeros(nbrItrMax)

		self.simulate()

		for itr in arange(avgVar):
			self.simulateBackward()
			self.simulate()
			previousOutput = self.output
			self.output = self.P_ase_f[:,-1]*(1-self.R2)
			outputError[i] = chi2(previousOutput, self.output)
			if verbose:
				print outputError[i]
			i += 1

		while (i < nbrItrMax):
			self.simulateBackward()
			self.simulate()
			previousOutput = self.output
			self.output = self.P_ase_f[:,-1]*(1-self.R2)
			outputError[i] = chi2(previousOutput, self.output)
			if verbose:
				print outputError[i]
			i += 1

		if errorOutput:
			return outputError





