# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:34:29 2015

@author: bxiao
"""

import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
from scipy.ndimage  import filters

img = misc.imread('sharpen_1.jpg',flatten=1)
kernel = (-1)*np.ones((3,3))
kernel[1,1] = 8

filtered = filters.convolve(img, kernel)
filtered1 = filtered - filtered.min()
filtered2 = filtered1 * (255.0/filtered.max())
#sharpened = img + filtered2
#show images
plt.figure()
plt.subplot(122)
plt.imshow(img, cmap = 'gray')
plt.subplot(121)
plt.imshow(filtered2, cmap = 'gray')
plt.show()