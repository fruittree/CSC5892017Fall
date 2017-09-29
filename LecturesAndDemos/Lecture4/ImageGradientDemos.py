# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 18:48:41 2015
Demostrates image derivatives
Using Gaussian filter, median filter to remove noise
@author: bxiao
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 22:33:16 2015
# Lecture 4: image dervatives
@author: bxiao
"""
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy import misc


l = misc.imread('jump.jpg',flatten=1)

gx_I,gy_I = np.gradient(l)[:2]

l = ndimage.gaussian_filter(l, 3)