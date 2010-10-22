import Image
import time
import fkabcd
import pyfits

from numpy import *
from pylab import *
from matplotlib.widgets import Slider, Button, RadioButtons


dz = 0.50/10
champ_in = Image.open("anneaux_large_ellipse2.gif")
intensiteArchive = zeros([512,512,512])
champ_in_fld = zeros([1024,1024], complex)
tampon = list(champ_in.getdata())


for x in range(1024):
	for y in range(1024):
		champ_in_fld[x,y] = float(tampon[(1024*y)+x])

# Premier step a 0.2
fkabcd.Prop(champ_in_fld, 632e-9, 0.0000116, 1.0, 0.2, 0.0, 1.0, 1.0 ,0.2, 0.0, 1.0)


# Step de 0.2 a 0.7
for i in range(10):
	fkabcd.Prop(champ_in_fld, 632e-9, 0.0000116, 1.0, dz, 0.0, 1.0, 1.0 ,dz, 0.0, 1.0)
	intensiteArchive[:,:,i] = pow(abs(champ_in_fld),2)[256:512+256,256:512+256]


ax = subplot(111)
subplots_adjust(left=0.25, bottom=0.25)
l, = imshow(intensiteArchive[:,:,1])


axcolor = 'lightgoldenrodyellow'
axfreq = axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
axamp  = axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0)
samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)

def update(val):
    amp = samp.val
    freq = sfreq.val
    l.set_ydata(amp*sin(2*pi*freq*t))
    draw()

sfreq.on_changed(update)
samp.on_changed(update)

resetax = axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
def reset(event):
    sfreq.reset()
    samp.reset()
button.on_clicked(reset)

rax = axes([0.025, 0.5, 0.15, 0.15], axisbg=axcolor)
radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)
def colorfunc(label):
    l.set_color(label)
    draw()
radio.on_clicked(colorfunc)

show()


