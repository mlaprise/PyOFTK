import PyOFTK
from pylab import *
from scipy import fftpack

T = 100
nt = pow(2,12)
dt = T/float(nt)
t = linspace(-T/2, T/2, nt)

z = 0.08
nz = 500
dz = z/nz

alpha = array([0])
betap = array([0,0,+1])

u_ini_x =  PyOFTK.gaussianPulse(t,2*sqrt(log(2)),0,1,1,0)
u_ini_y = zeros(nt)

u_out_x = PyOFTK.ssf(u_ini_x, dt, dz, nz, alpha, betap, pow(30,2),50, 1e-5)
[WL,U] = PyOFTK.pulseSpectrum(t, u_out_x)

figure(figsize=(15,6))
subplot(121)
plot(t,pow(abs(u_out_x),2), color='blue')
xlim([-5,5])
xlabel (r"($t-\beta_1z)/T_0$")
ylabel (r"$|u(z,t)|^2/P_0$")

subplot(122)
plot(WL,U, color='blue')
xlabel (r"$(\nu-\nu_0)T_0$")
#xlabel (r"Walelength [nm]")
ylabel (r"$|U(z,\nu)|^2/P_0$")
xlim([-10,10])

show()
