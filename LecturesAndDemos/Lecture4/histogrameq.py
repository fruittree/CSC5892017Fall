# histogram equalization
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

def histeq(im,nbr_bins=256):
    """ Histogram equalization of a grayscale image. """

    
# conver to grayscale    
#im = np.array(Image.open("lowcontrast.jpg").convert("L"))
im = misc.imread('lowcontrast.jpg',flatten=1)

print int(im.min()), int(im.max())
im2,cdf = histeq(im)

# compute the histogram of the new image


# plot the before and after histograms
# plot the cdf


plt.plot(cdf_normalized, color = 'b')
plt.hist(im2.flatten(),256,[0,256], color = 'r')
#plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

