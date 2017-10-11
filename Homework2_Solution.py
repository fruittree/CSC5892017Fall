'''
Solution

'''
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import scipy
from scipy import misc
from scipy import ndimage
from fractions import Fraction
import math
from scipy.ndimage import filters

# ===  Problem 1: Warm Up ===
# 1.1 loading the images
#
image1 = np.float64(misc.imread('images/peppers.png', flatten=1, mode='F'))
image2 = np.float64(misc.imread('images/cheetah.png', flatten=1, mode='F'))


# 1.2 blur the images
#
gau_image1 = ndimage.gaussian_filter(image1, 7)
gau_image2 = ndimage.gaussian_filter(image2, 7)


# 1.3 display the image
#
plt.figure(1, figsize=(15, 5))
plt.suptitle('problem 1.3', fontsize=20, fontweight='bold')

plt.subplot(1, 4, 1)
plt.title('image1', fontsize=10)
plt.imshow(image1, cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(1, 4, 2)
plt.title('image1_blurred', fontsize=10)
plt.imshow(gau_image1, cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(1, 4, 3)
plt.title('image2', fontsize=10)
plt.imshow(image2, cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(1, 4, 4)
plt.title('image1_blurred', fontsize=10)
plt.imshow(gau_image2, cmap=plt.cm.gray)
plt.axis('off')
plt.show()



# 1.4 compute dft of the image
#
# image1
dft_image1 = np.fft.fft2(image1)
dft_image1_shift = np.fft.fftshift(dft_image1)
magnitude_spectrum_image1 = np.log(np.abs(dft_image1_shift))
# image2
dft_image2 = np.fft.fft2(image2)
dft_image2_shift = np.fft.fftshift(dft_image2)
magnitude_spectrum_image2 = np.log(np.abs(dft_image2_shift))


# plot
plt.figure(1, figsize=(15, 5))
plt.suptitle('problem 1.4', fontsize=20, fontweight='bold')

plt.subplot(1, 2, 1)
plt.title('dtf_image1', fontsize=10)
plt.imshow(magnitude_spectrum_image1, cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title('dtf_image2', fontsize=10)
plt.imshow(magnitude_spectrum_image2, cmap=plt.cm.gray)
plt.axis('off')
plt.show()



# ===Problem 2: Histogram equilization ===

im2 = np.float64(misc.imread('images/lowcontrast.jpg', flatten=1))
# histogram
frequencies, bins = numpy.histogram(im2, bins=numpy.arange(-0.5, 255.1, 0.5))

intensities = bins[1:bins.size]
# cdf
cdf = np.cumsum(frequencies)
# transfer function
intensities_mapping = cdf/np.float32(cdf[-1]) * 255


# truncate
target_number_of_pure_blackwhite = cdf[-1] * 0.025
# map to pure black
number_of_black = 0
i = 0
while (number_of_black < target_number_of_pure_blackwhite):
    number_of_black += frequencies[i]
    intensities_mapping[i] = 0
    i += 1
# map to pure white
number_of_white = 0
j = frequencies.size - 1
while (number_of_white < target_number_of_pure_blackwhite):
    number_of_white += frequencies[j]
    intensities_mapping[j] = 255
    j -= 1

counts = im2.size - (number_of_black + number_of_white)


# equalize
tolerance = counts*0.025
trimmings = 0
while(trimmings < tolerance):
    trimmings = 0
    counts = np.sum(frequencies[i:(j+1)])
    for k in range(i, j+1):
        ceiling = counts/(j+1-i)
        if frequencies[k] > ceiling:
            trimmings += frequencies[k] - ceiling
            frequencies[k] = ceiling

intensities_mapping[i:(j+1)] = np.cumsum(frequencies[i:(j+1)]).astype(float)/counts*255.0 + 255*(float(number_of_black)/im2.size)
im2_he = np.interp(im2, intensities, intensities_mapping)

plt.figure()
plt.title('Histogram-equalized', fontsize=10)
plt.imshow(im2_he, cmap=plt.cm.gray)
plt.show()



# === Problem 3:  Separable filters ===
im4 = np.float64(misc.imread('images/einstein.png', flatten=1))

gaussian_kernel = 1.0/256*numpy.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]])
gaussian_kernel_x = numpy.array([1,4,6,4,1])
gaussian_kernel_y = 1.0/256*numpy.array([1,4,6,4,1])

box_kernel = numpy.ones((5,5), dtype=numpy.float32)/25
box_kernel_x = numpy.array([1,1,1,1,1])
box_kernel_y = numpy.array([1,1,1,1,1])/25.0

sobel_kernel = numpy.array([[1,2,0,-2,-1],[4,8,0,-8,-4],[6,12,0,-12,-6],[4,8,0,-8,-4],[1,2,0,-2,-1]])
sobel_kernel_x = numpy.array([1,2,0,-2,-1])
sobel_kernel_y = numpy.array([1,4,6,4,1])

im4_gaussian = filters.convolve(im4, gaussian_kernel, mode="mirror")
im4_gaussian_x = filters.convolve1d(im4, gaussian_kernel_x, axis=1, mode = "mirror")
im4_gaussian_y = filters.convolve1d(im4, gaussian_kernel_y, axis=0, mode = "mirror")

im4_box = filters.convolve(im4, box_kernel, mode="wrap") 
im4_box_x = filters.convolve1d(im4, box_kernel_x, axis=1, mode="wrap")
im4_box_y = filters.convolve1d(im4, box_kernel_y, axis=0, mode="wrap")

im4_sobel = filters.convolve(im4, sobel_kernel, mode="nearest")
im4_sobel_x = filters.convolve1d(im4, sobel_kernel_x, axis=1, mode="nearest")
im4_sobel_y = filters.convolve1d(im4, sobel_kernel_y, axis=0, mode="nearest")

fig4_gaussian = plt.figure()
fig4_gaussian.suptitle("Gaussian convolution")
im4_plot = fig4_gaussian.add_subplot(2,2,1)
im4_plot.set_title("Original")
im4_plot.imshow(im4, cmap=plt.cm.gray)
im4_gaussian_plot = fig4_gaussian.add_subplot(2,2,2)
im4_gaussian_plot.set_title("Gaussian-blur")
im4_gaussian_plot.imshow(im4_gaussian, cmap=plt.cm.gray)
im4_gaussian_x_plot = fig4_gaussian.add_subplot(2,2,3)
im4_gaussian_x_plot.set_title("Horizontal Gaussian")
im4_gaussian_x_plot.imshow(im4_gaussian_x, cmap=plt.cm.gray)
im4_gaussian_y_plot = fig4_gaussian.add_subplot(2,2,4)
im4_gaussian_y_plot.set_title("Vertical Gaussian")
im4_gaussian_y_plot.imshow(im4_gaussian_y, cmap=plt.cm.gray)


fig4_box = plt.figure()
fig4_box.suptitle("Box Convolution")
im4_plot = fig4_box.add_subplot(2,2,1)
im4_plot.set_title("Original")
im4_plot.imshow(im4, cmap=plt.cm.gray)
im4_box_plot = fig4_box.add_subplot(2,2,2)
im4_box_plot.set_title("Box-blur")
im4_box_plot.imshow(im4_box, cmap=plt.cm.gray)
im4_box_x_plot = fig4_box.add_subplot(2,2,3)
im4_box_x_plot.set_title("Horizontal box")
im4_box_x_plot.imshow(im4_box_x, cmap=plt.cm.gray)
im4_box_y_plot = fig4_box.add_subplot(2,2,4)
im4_box_y_plot.set_title("Vertical box")
im4_box_y_plot.imshow(im4_box_y, cmap=plt.cm.gray)


fig4_sobel = plt.figure()
fig4_sobel.suptitle("Sobel convolution")
im4_plot = fig4_sobel.add_subplot(2,2,1)
im4_plot.set_title("Original")
im4_plot.imshow(im4, cmap=plt.cm.gray)
im4_sobel_plot = fig4_sobel.add_subplot(2,2,2)
im4_sobel_plot.set_title("Sobel")
im4_sobel_plot.imshow(im4_sobel, cmap=plt.cm.gray)
im4_sobel_x_plot = fig4_sobel.add_subplot(2,2,3)
im4_sobel_x_plot.set_title("Horizontal sobel")
im4_sobel_x_plot.imshow(im4_sobel_x, cmap=plt.cm.gray)
im4_sobel_y_plot = fig4_sobel.add_subplot(2,2,4)
im4_sobel_y_plot.set_title("Vertical sobel")
im4_sobel_y_plot.imshow(im4_sobel_y, cmap=plt.cm.gray)

plt.show()





# === Problem 4 ===
grascale = np.float64(misc.imread('images/zebra.png', flatten=1, mode='F'))
edge_horizont = ndimage.sobel(grascale, 0)
edge_vertical = ndimage.sobel(grascale, 1)
magnitude = np.hypot(edge_horizont, edge_vertical)


plt.figure(figsize=(10, 5))
plt.suptitle('problem 4: edge detection', fontsize=20, fontweight='bold')
plt.subplot(1, 4, 1)
plt.imshow(grascale, cmap=plt.cm.gray)
plt.axis('off')
plt.title('original')
plt.subplot(1, 4, 2)
plt.imshow(edge_horizont, cmap=plt.cm.gray)
plt.title('x-axis edges')
plt.axis('off')
plt.subplot(1, 4, 3)
plt.imshow(edge_vertical, cmap=plt.cm.gray)
plt.title('y-axis edges')
plt.axis('off')
plt.subplot(1, 4, 4)
plt.imshow(magnitude, cmap=plt.cm.gray)
plt.title('all edges')
plt.axis('off')
plt.show()
