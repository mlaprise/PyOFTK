import PyOFTK
from pylab import *


T = 48.0
nt = pow(2,10)
dt = T/nt
t = linspace(-T/2, T/2, nt)
dz = 5.0

u_ini_x = exp(-pow(t,2)/2)
u_ini_y = zeros(nt)
betap = array([0,0,0,1])
alpha = array([0])

u_out_x = PyOFTK.ssf(u_ini_x, dt, dz, 1, alpha, betap, 0.000,4,1e-5)

betap = array([0,0,1,1])
u_out2_x = PyOFTK.ssf(u_ini_x,dt, dz, 1, alpha, betap, 0.000, 4,1e-5)

plot(t, pow(abs(u_out_x),2),t, u_ini_x, ':',t, pow(abs(u_out2_x),2), '--',color='black')
ylabel("$|u(z,T)|^2$")
xlabel("$T/T_0$")
legend((r"$\beta_2 = 0$", "$z = 0$","$L_D = L_D'$"))
xlim([-12,12])
show()
