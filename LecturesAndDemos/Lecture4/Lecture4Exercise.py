# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 22:33:16 2015
# Lecture 4: image dervatives exercises
# Some of the code was given to you. Please finish the rest. 
@author: bxiao
"""

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# 1. generate an image with rotated rectangle  
im = np.zeros((256, 256))
im[64:-64, 64:-64] = 1
im = ndimage.rotate(im, 15, mode='constant')
print im
plt.imshow(im,cmap=plt.cm.gray)
plt.show()
  
# 2 Blur it with a gaussian filter


# 3. convolve with sobel filter to both x and y direciton.   
#  can use np.hypot to compute the magnitute. 
  
# 4. plot the original image, x-derivatives and y-derivatives. 



plt.figure(figsize=(20, 4))
plt.subplot(141)
# original
plt.imshow(im, cmap=plt.cm.gray)
plt.axis('off')
plt.title('square', fontsize=20)
plt.subplot(142)
# sobel x 
plt.imshow(sx)
plt.axis('off')
plt.title('Sobel (x direction)', fontsize=20)
plt.subplot(143)
plt.imshow(sy)
plt.axis('off')
plt.title('Sobel (y direction)', fontsize=20)
plt.subplot(144)
plt.imshow(sob)
plt.axis('off')
plt.title('magnitude', fontsize=20)
   
plt.show()