from scipy.ndimage import filters
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
# read in an image using mis. 

im = misc.imread('shadow.png',flatten=1)
print int(im.min()), int(im.max())

# resize the image
#im_resized = misc.imresize(im, (281,211), interp='bilinear', mode=None)
#plt.figure()
#plt.imshow(im, vmin = 0, vmax = 256,cmap=plt.cm.gray)
#plt.show()

# Blur with Gaussian filter
#im2 = np.zeros(im.shape)
im_blur = filters.gaussian_filter(im,3)
misc.imsave('baby_blur.png', im_blur) 
plt.figure()
plt.imshow(im_blur,cmap=plt.cm.gray)
plt.show()

# Sharpen a blurred image (unsharp masking)
im_blur = filters.gaussian_filter(im, 7)
# blur the blurre image
im_blur2 = filters.gaussian_filter(im_blur,3)
alpha =10
im_sharpened = im_blur + alpha * (im_blur - im_blur2)

# plotting3 figures in one subplot
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.imshow(im, cmap=plt.cm.gray, vmin = 0, vmax = 255)
plt.subplot(132)
plt.imshow(im_blur,cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(133)
plt.imshow(im_sharpened, cmap=plt.cm.gray)
plt.axis('off')
plt.show()
#==============================================================================
# # take home exercises: 
# # can you write your own function unsharp_mask that takes a blurred image and make a sharper?
# define gaussian kernal
def gaussian_kernel(size, size_y=None):
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    x, y = np.mgrid[-size:size+1, -size_y:size_y+1]
    g = np.exp(-(x**2/float(size)+y**2/float(size_y)))
    return g / g.sum()

# customed unsharp_mask filter
def unsharpfilter(kernal_size,alpha):
	# define the filter. 
   	g_kernal = gaussian_kernel(kernal_size)
   	sigma = np.eye(g_kernal.shape[1])
   	# compute the filter
   	filter_us = g_kernal + alpha*sigma-alpha*g_kernal
   	return filter_us

# convolve the image with the filter
filter_us = unsharpfilter(3,10)
sharpened_image = filters.convolve(im, filter_us, mode='constant', cval=0.0)

# Plotting 
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(im_blur, cmap=plt.cm.gray, vmin = 0, vmax = 255)
plt.subplot(122)
plt.imshow(sharpened_image ,cmap=plt.cm.gray)
plt.axis('off')
plt.show()




# # Can you write a filter that is unsharpmask? 
# That you only have to convolve the image once! See lecture notes for details. 
# 
# 
# 
# # plot the image histogram, cdf and normalized cdf here: 
#==============================================================================




