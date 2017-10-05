# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 20:40:57 2015

@author: bxiao
"""

import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

# read image
img = plt.imread('cheetah.png')

# prepare an 1-D Gaussian convolution kernel
t = np.linspace(-10, 10, 30)
bump = np.exp(-0.1*t**2)
bump /= np.trapz(bump) # normalize the integral to 1

# make a 2-D kernel out of it
kernel = bump[:, np.newaxis] * bump[np.newaxis, :]

scharr = np.array([[-3, 0, 3],
                   [-10,0,10],
                   [-3, 0, 3]])

sobel_x= np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

sobel_y= np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])

kernel = sobel_y

# padded fourier transform, with the same shape as the image
kernel_ft = fftpack.fft2(kernel, shape=img.shape[:2], axes=(0, 1))
plt.imshow(np.abs(kernel_ft),cmap='gray')
plt.show()

# FFT of the original image
#img_ft = fftpack.fft2(img, axes=(0, 1))
img_ft = np.fft.fft2(img)
img_ft= np.fft.fftshift(img_ft )
# convolve the orignal image

img2_ft = kernel_ft[:, :, np.newaxis] * img_ft

img2 = fftpack.ifft2(img2_ft, axes=(0, 1)).real

# clip values to range
img2 = np.clip(img2, 0, 1)

# plot output
plt.imshow(np.abs(img2))
plt.show()
