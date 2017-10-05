# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 23:20:30 2015
@author: bxiao
"""

from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

# read in an image using mis. 
img = misc.imread('tree.jpg',flatten=1)
# discreet foureir transform in Numpy
# https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.fft.html
# np.fft.fft2 compute the 2-dimensional discrete Fourier Transform

f = np.fft.fft2(img)
# Shift the zero-frequency component to the center of the spectrum.
fshift = np.fft.fftshift(f)

magnitude_spectrum = np.log(np.abs(fshift))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Fabric Texture'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Log Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()


# let's look at a noisy image 
pepper = misc.imread('peppers256.png',flatten=1)
f = np.fft.fft2(pepper)
fshift = np.fft.fftshift(f)
magnitude_spectrum_p = np.log(np.abs(fshift))

pepper_noise = misc.imread('peppers256_noisy.png',flatten=1)
f = np.fft.fft2(pepper_noise)
fshift = np.fft.fftshift(f)
magnitude_spectrum_pn = np.log(np.abs(fshift))


plt.subplot(121),plt.imshow(magnitude_spectrum_p, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum_pn, cmap = 'gray')
plt.title('Noisey'), plt.xticks([]), plt.yticks([])
plt.show()




