# histogram equalization
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

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
    
# conver to grayscale    
#im = np.array(Image.open("lowcontrast.jpg").convert("L"))
im = misc.imread('lowcontrast.jpg',flatten=1)

print int(im.min()), int(im.max())
im2,cdf = histeq(im)

# compute the histogram of the new image
imhist2,bins = np.histogram(im2.flatten(),256,[0,256])
cdf2 = imhist2.cumsum() 
cdf_normalized = cdf2 * imhist2.max()/ cdf2.max()

#
imhist,bins = np.histogram(im.flatten(),256,[0,256])
cdf = imhist.cumsum() 
cdf_normalized1 = cdf * imhist2.max()/ cdf.max()

plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.imshow(im, vmin = 0, vmax = 256,cmap=plt.cm.gray)
plt.subplot(122)
plt.imshow(im2, vmin = 0, vmax = 256,cmap=plt.cm.gray)
plt.axis('off')
plt.show()

# plot the before and after histograms
# plot the cdf

plt.plot(cdf_normalized1, color = 'b')
plt.hist(im.flatten(),256,[0,256], color = 'r')
#plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()

plt.plot(cdf_normalized, color = 'b')
plt.hist(im2.flatten(),256,[0,256], color = 'r')
#plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()




