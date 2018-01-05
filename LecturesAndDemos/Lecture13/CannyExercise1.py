import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

from skimage.feature import canny
import cv2

# Exercise 1
# Generate noisy image of a square
im = np.zeros((128, 128))
im[32:-32, 32:-32] = 1

im = ndimage.rotate(im, 15, mode='constant')
im = ndimage.gaussian_filter(im, 4)
im += 0.2 * np.random.random(im.shape)

# use skimage.feature.canny
# to extract edges of the above images by adjusting sigma values. 


# Exercise 2
# Read in your favourite image as .jpg.
# Using opencv's cv2.canny to perform edge detection.
# Experiment with how low/high threshold change the extracted edges


# here are some plotting routines 

# you have to define edges1 edges2 yourself. 

# display results
# fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))

# ax1.imshow(im, cmap=plt.cm.jet)
# ax1.axis('off')
# ax1.set_title('noisy image', fontsize=20)

# ax2.imshow(edges1, cmap=plt.cm.gray)
# ax2.axis('off')
# ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)

# ax3.imshow(edges2, cmap=plt.cm.gray)
# ax3.axis('off')
# ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)

# fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
#                     bottom=0.02, left=0.02, right=0.98)

# plt.show()






