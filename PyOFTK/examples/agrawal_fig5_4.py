from numpy import *
from PyOFTK.utilities import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import PyOFTK
import scipy.signal as ss

plotRes = 128	 
T = 10.0
nt = pow(2,12)
dt = T/nt
t = linspace(-T/2.0, T/2.0, nt)

z = pi/2
nz = 500.0
nplot = plotRes
n1 = round(nz/nplot)
nz = n1*nplot
dz = z/nz

betap = array([0,0,-1])
alpha = array([0.0])

zv = (z/nplot)*arange(0,nplot)
u = zeros([len(t),len(zv)], complex)
uplot = zeros([plotRes,len(zv)], float32)

u[:,0] = PyOFTK.solitonPulse(t,0,1.0,3.0)
uplot[:,0] = ss.resample(pow(abs(u[:,0]),2), plotRes)

for ii in arange(0,nplot-1):
	#u[:,ii+1] = PyOFTK.ssf(u[:,ii], dt, dz, n1, 0.0, betap, 1.0, 10, 1e-5)
	u[:,ii+1] = PyOFTK.scalar(u[:,ii], dt, dz, n1, alpha, betap, 1.0)
	uplot[:,ii+1] = ss.resample(pow(abs(u[:,ii+1]),2), plotRes)

tplot= linspace(-T/2.0, T/2.0, plotRes)
X, Y = meshgrid(zv, tplot)
fig = plt.figure()
ax = axes3d.Axes3D(fig)
ax.plot_wireframe(X, Y, uplot)
ax.set_xlabel (r"$Z/Z_0$");
ax.set_ylabel (r"($t-\beta_1z)/T_0$");
ax.set_zlabel (r"$|u(z,t)|^2/P_0$");
plt.show()

