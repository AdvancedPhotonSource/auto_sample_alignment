### TODO ###
### source activate alignment
import os
import sys
import numpy
import scipy
import math
import logging
import time
import rlcompleter
import readline

import datetime as dt
import epics as PyEpics
import matplotlib.pyplot as plt


from scipy.special import erf
from scipy.optimize import curve_fit
from pprint import pprint
from mpl_toolkits.mplot3d import axes3d

### PIN TRACKING PACKAGE
# sys.path.append('/home/beams/S1IDUSER/auto_sample_alignment/src')
from align import Alignment


def tracking(params, algorithm='pin', repeat=3):
    
    #### ENABLE TAB COMPLETION
    readline.parse_and_bind('tab: complete')

    #################################################
    pixel_size = 1.172; ## um / pixel

    slit_center_x=1000;
    slit_center_y=600;

    #################################################
    ### MOTOR SETTINGS
    #################################################
    mtr_samXE = PyEpics.Motor('1ide1:m34');
    mtr_samYE = PyEpics.Motor('1ide1:m35');
    mtr_samZE = PyEpics.Motor('1ide1:m36');
    mtr_samTh = PyEpics.Motor('1ide1:m86');
    mtr_samChi = PyEpics.Motor('1ide1:m87');
    mtr_samOme = PyEpics.Motor('1ide:m9');
    mtr_aeroXE = PyEpics.Motor('1ide1:m101');
    mtr_aeroZE = PyEpics.Motor('1ide1:m102');
    #################################################

    pname='/home/beams/S1IDUSER/mnt/s1c/mli_nov19/tomo';

    

    for i in range(repeat):
        #################################################
        ### ALIGN ROTATION AXIS TO THE HS
        #################################################
        ### MOVE STAGE TO 0
        mtr_samOme.move(0, wait=True)

        ### TAKE AN IMAGE AT OME 0
        PyEpics.caput('1idPG1:cam1:Acquire', 1)
        time.sleep(3)

        fname=PyEpics.caget('1idPG1:TIFF1:FileName_RBV', 'str') + "_%06d"%(PyEpics.caget('1idPG1:TIFF1:FileNumber_RBV')-1) + '.tif'
        pfname=os.path.join(pname, fname)

        print(pfname)
    
        align0 = Alignment(pfname)
        x0, y0 = align0.compute_center(algorithm, params)
        print(f"pin x,y position when omega is 0 : (x = {x0}, y = {y0}")
        
        if (x0 == -1) or (y0 == -1):
            print("Alignment failed at zero degress")
            break

        ### MOVE STAGE TO 180
        mtr_samOme.move(180, wait=True)

        ### TAKE AN IMAGE AT OME 180
        PyEpics.caput('1idPG1:cam1:Acquire', 1)
        time.sleep(3)

        fname=PyEpics.caget('1idPG1:TIFF1:FileName_RBV', 'str') + "_%06d"%(PyEpics.caget('1idPG1:TIFF1:FileNumber_RBV')-1) + '.tif'
        pfname=os.path.join(pname, fname)

    
        align180 = Alignment(pfname)
        x180, y180 = align180.compute_center(algorithm, params)
        if (x180 == -1) or (y180 == -1):
            print("Alignment failed at 180 degrees")
            break
            
        print(f"pin x,y position when omega is 180 : (X = {x180}, y = {180})")
    
        ### COMPUTE MOTIONS TO MAKE
        mid_x = (x180 + x0)/2;
        half_delta_x = (x180 - x0)/2;

        print(mid_x)
        print(half_delta_x)

        ### NEED TO CHECK / FIGURE OUT THE SIGNS ON THESE 
        aeroXE_motion = ((mid_x - slit_center_x)*pixel_size)/1000;
        samXE_motion = -(half_delta_x*pixel_size)/1000;

        print('motions to execute')
        print(aeroXE_motion)
        print(samXE_motion)

        mtr_aeroXE.move(aeroXE_motion, relative=True, wait=True)
        mtr_samXE.move(samXE_motion, relative=True, wait=True)
