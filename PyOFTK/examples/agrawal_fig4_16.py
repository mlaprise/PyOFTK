from numpy import *
from pylab import *
from PyOFTK.utilities import *
import PyOFTK

T = 16.0
nt = pow(2,10)
dt = T/nt
z = 20.0
nz = 1000
nplot = 2
n1 = round(nz/nplot)
nz = n1*nplot
dz = z/nz

t = linspace(-T/2.0, T/2.0, nt)

s = 0.01

zv = (z/nplot)*(0:nplot);
u = zeros(length(t),length(zv))
u_ini_y = zeros(nt)
U = zeros(length(t),length(zv))

u[:,1] = PyOFTK.gaussian(t,0,2*(sqrt(log(2.0))))
U[:,1] = fftshift(abs(dt*fft(u[:,1])/sqrt(2*pi)).^2)


for ii in arange(1:nplot)
	u[:,ii+1] = PyOFTK.scalar(u[:,ii],dt,dz,n1,0,0,1,0,2*pi*s)
	U[:,ii+1] = fftshift(abs(dt*fft(u[:,ii+1])/sqrt(2*pi)).^2)


plot (t,pow(abs(u),2))
xlim([-3,3])
grid(True)
xlabel ('(t-\beta_1z)/T_0')
ylabel ('|u(z,t)|^2/P_0')
show()
