# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 16:15:29 2015
@author: bxiao
"""
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import cv2

# read in an image using mis. 
img = misc.imread('einstein.png',flatten=1)


f = np.fft.fft2(img)
# fshift the FFT image.
fshift = np.fft.fftshift(f)
magnitude_spectrum = 30*np.log(np.abs(fshift))

# display the original image and FFT
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Log Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

# removing the high frequencies 
rows, cols = img.shape
crow, ccol = rows/2 , cols/2     # center
# create a mask first, center square is 1, remaining all zeros
#mask = np.zeros((rows, cols), np.uint8)
#mask[crow-30:crow+30, ccol-30:ccol+30] = 1
#plt.imshow(mask,cmap='gray')
#plt.show()
#print mask.shape

# apply the mask: this is the same as convolution in spacial domain!! 

# removing low frequency, high pass
# There are many ways to create the mask
#fshift[crow-5:crow+5, ccol-5:ccol+5] = 1
#fshift[crow-120:crow-5, ccol+5:ccol+160] = 10
#fshift[crow-120:crow-5, ccol-120:ccol-5] = 10
#fshift[crow+5:crow+120, ccol-120:ccol-5] = 10
# thresholding the values
# print fshift.max(), fshift.min()
# mask = abs(fshift) < 10000
# plt.imshow(mask)
# #plt.show()
# #plt.show()
# ##print mask
# fshift[mask]= 100

fshift[crow-3:crow+3, ccol-3:ccol+3] = 0
print fshift.max(),fshift.min()
magnitude_spectrum2 = 30*np.log(np.abs(fshift))
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

print int(img_back.min()), int(img_back.max())
# histogram equalization
def histeq(im,nbr_bins=256):
    """ Histogram equalization of a grayscale image. """
    # get image histogram
    imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True)
    # cumulative distribution function
    cdf = imhist.cumsum() 
    # normalize
    cdf = 255 * cdf / cdf[-1] 
    # use linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape), cdf    
#this image needs to be normalized.
img_back2,cdf = histeq(img_back)

 
plt.subplot(131),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Input Image after FFT'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(magnitude_spectrum2 , cmap = 'gray')
plt.title('New FFT Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(img_back2, cmap='gray')
plt.title('Image after low-pass filter'), plt.xticks([]), plt.yticks([])
plt.show()                

## Exercise 1: how do you remove the low frequency?  Display the DFT magnitutde and the resulting image here




