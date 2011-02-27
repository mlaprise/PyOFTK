PyOFTK: Python Optical Fiber ToolKit
++++++++++++++++++++++++++++++++++++

PyOFTK is a Python/Numpy library for helping the developement of numerical simulation of optical pulses in optical fiber. PyOFTK is a work-in-progress project and for now it’s more like a whole bunch of functions and class I made for my own use during my graduate research.


Here is a simple example how we can use PyOFTK in iPython::

	In [1]: import PyOFTK

	In [2]: fiber1 = PyOFTK.FibreStepIndex(3.0, 62.5, 0.05, 0.0, 1.0)

	In [3]: fiber1.printLPModes(1.03)
	Mode LP01: u=1.71279090763
	Mode LP02 unsupported
	Mode LP03 unsupported
	Mode LP04 unsupported
	Mode LP05 unsupported
	Mode LP11: u=2.58518963818

	In [4]: print "Dispersion: " + str(fiber1.beta2(1.03) * 1e24) + " ps2/m"
	Dispersion: 0.0215828748536 ps2/m

	In [5]: fiber1.effIndex(1.03)       
	Out[5]: 1.4544886960584553

	In [11]: fiber1.modeOverlap(1.03)  
	Out[11]: 0.6149059753294210


Installation
============

Requirements
------------
	* FFTW 3
	* FFTW 2 (for the freespace module)
	* PyCUDA 0.94 or newer
	* Numpy 1.1.1 or newer
	* Scipy 0.6.0 or newer
	* Matplotlib 0.98
	* python-vtk 5.0
	* pyfits 1.3
	* python-tables 2.0.3
	* Swig 1.3
