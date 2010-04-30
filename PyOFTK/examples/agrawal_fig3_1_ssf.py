from numpy import *
from pylab import *
from PyOFTK.utilities import *
import PyOFTK


T = 32.0
nt = pow(2,10)
dt = T/nt
t = linspace(-T/2.0, T/2.0, nt)
dz = 2.0
nz = 2

betap = array([0.0,0.0,1.0])
alpha = 0.0
u_ini_x = PyOFTK.gaussianPulse(t,2.0,0.0,1.0,1.0,0.0)
u_out_x = PyOFTK.ssf(u_ini_x, dt, dz, nz, alpha, betap, 0.0, 10, 1e-5)

plot(t, pow(abs(u_out_x),2),t, pow(abs(u_ini_x),2))
ylabel("$|u(z,T)|^2$")
xlabel("$T/T_0$")
xlim([-16,16])
grid(True)
show()

