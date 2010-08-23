#!/usr/bin/env python

import Image
import time
import fkabcd
import pyfits
import pylab as pl
from numpy import *


step = 10
dz = 0.50/step
champ_in = Image.open("anneaux_mince.gif")
intensiteArchive = zeros([512,512,step])
champ_in_fld = zeros([1024,1024], complex)
tampon = list(champ_in.getdata())


for x in range(1024):
	for y in range(1024):
		champ_in_fld[x,y] = float(tampon[(1024*y)+x])

t1 = time.time()

# Premier step a 0.2
fkabcd.Prop(champ_in_fld, 632e-9, 0.0000116, 1.0, 0.2, 0.0, 1.0, 1.0 ,0.2, 0.0, 1.0)

# Step de 0.2 a 0.7
for i in range(step):
	fkabcd.Prop(champ_in_fld, 632e-9, 0.0000116, 1.0, dz, 0.0, 1.0, 1.0 ,dz, 0.0, 1.0)
	intensiteArchive[:,:,i] = pow(abs(champ_in_fld),2)[256:512+256,256:512+256]

# Bench
elapsed = time.time() - t1
print "Processing time: " + str(elapsed) + " secondes" + "(" + str(elapsed/60) + " minutes)"

pl.imshow(pow(abs(champ_in_fld),2), aspect='equal', interpolation='bicubic')
pl.show()



