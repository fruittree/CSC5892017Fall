# -*- coding: utf-8 -*-
"""
# Basic image tutorials
Created on Tue Jan 13 20:47:26 2015

@author: bxiao
"""
import numpy as np
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from mpl_toolkits.mplot3d import Axes3D

##  load and display an image
baby = misc.imread('Kaila.jpg',flatten=1)
plt.imshow(baby,cmap=plt.cm.gray)
plt.show()

# here is the code to resize a big image, it is not effective in the lena case
#baby_resized = misc.imresize(baby, (512,512), interp='bilinear', mode=None)
#plt.imshow(baby_resized,cmap=plt.cm.gray)
#plt.show()

# print out some information 
print baby.shape
print baby.dtype
print baby.max()
print baby.min()

# change brightness
# darker

# baby_dark = baby-125
# plt.imshow(baby_dark, vmin = 0, vmax = 128,cmap=plt.cm.gray)
# plt.show()
# misc.imsave('baby_dark.png', baby_dark) 

# #
# #
# ## create a surface plot of the image
# #x, y = np.ogrid[0:baby.shape[0], 0:baby.shape[1]]
# #fig=plt.figure()
# #ax = Axes3D(fig)
# #ax.plot_surface(x,y,baby,rstride=4, cstride=4, cmap=plt.cm.jet, linewidth=0.2)
# #plt.show()


# Image convolution
# mean filter
# there are many ways to do convolution in Python.
# I use scipy.ndimage.filters.convolve
# Other choices are: scipy.signal.convolve2d
k = np.ones((5,5))/25
# convolve with the image
b= ndimage.filters.convolve(baby,k)
b = misc.toimage(b)
b.save('baby_blur.png')
plt.imshow(b, cmap=plt.cm.gray)
plt.show()
#
#
## or you can use uniform_filter, same as filters.convolve with a uniform kernel
local_mean = ndimage.uniform_filter(baby, size=5)
plt.imshow(local_mean, cmap=plt.cm.gray)
plt.show()

# #
## sharpening a blurred image
blurred = ndimage.gaussian_filter(baby, 5)
print blurred
plt.imshow(blurred, cmap=plt.cm.gray)
plt.show()

filter_blurred= ndimage.gaussian_filter(blurred,1)
alpha = 30
sharpened = blurred + alpha * (blurred - filter_blurred)
#
# plotting3 figures in one subplot
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(baby, cmap=plt.cm.gray, vmin = 0, vmax = 255)
plt.subplot(132)
plt.imshow(blurred,cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(133)
plt.imshow(sharpened, cmap=plt.cm.gray)
plt.axis('off')

plt.show()
#
#
# #==============================================================================
# # ### cropping
# # ##lx,ly = lena.shape
# # ##crop_lena = lena[lx / 4: - lx / 4, ly / 4: - ly / 4]
# # ##flip_ud_lena = np.flipud(lena)
# # ##rotate_lena = ndimage.rotate(lena, 45)
# #==============================================================================

# #==============================================================================
#  ## generate image from noise
# #plt.figure
# #im = np.zeros((128, 128))
# #im[32:-32, 32:-32] = 1
# #im = ndimage.rotate(im, 15, mode='constant')
# #im = ndimage.gaussian_filter(im, 4)
# #im += 0.2 * np.random.random(im.shape)
# #plt.imshow(im, cmap=plt.cm.jet)
# #plt.show()
#==============================================================================




