# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 15:46:53 2015
image histogram manually 
@author: bxiao
"""
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

# read in the image 
im = misc.imread('lowcontrast.jpg',flatten=1)
plt.imshow(im, vmin = 0, vmax = 256,cmap=plt.cm.gray)
plt.show()

# get the histogram
hist,bins = np.histogram(im.flatten(),256,[0,256])

# compute the cdf
cdf = hist.cumsum()
# normalize the cdf
cdf_normalized = cdf * hist.max()/ cdf.max()


plt.plot(cdf, color = 'b')
plt.hist(im.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

# plot the cdf
plt.plot(cdf_normalized, color = 'b')
plt.hist(im.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

# normalize the cdf
cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')

# now compute the new histogram and cdf as the above code



