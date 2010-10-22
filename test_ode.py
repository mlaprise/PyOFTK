#!/usr/bin/env python
__version__ = "0.01"
__date__ = "2007-07-19"
__author__ ="Martin Laprise"


# *********************************************
#	Test de la fonction ODE de scipy avec
#	l'example 7.12 du livre 'Analyse numerique
#	pour ingenieurs
#
#
#	Author: 	Martin Laprise
#		    	Universite Laval
#				martin.laprise.1@ulaval.ca
#                 
# *********************************************


from scipy import *
from scipy import integrate
from pylab import *


systemeEquation = lambda y,t : array([y[1],2*y[1]-y[0]])


# Integration parameters
debut=0
fin=2
numsteps=10000
time=linspace(debut,fin,numsteps)

#
y0=array([2,1])
y=integrate.odeint(systemeEquation,y0,time)
plot(time,y[:,0],time,y[:,1])
title('A. Fortin: Exemple 7.12')
grid(True)
show()
