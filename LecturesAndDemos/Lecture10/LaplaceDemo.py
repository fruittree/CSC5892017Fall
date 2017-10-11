from scipy import ndimage, signal,misc
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
plt.gray()  # show the filtered result in grayscale

ax1 = fig.add_subplot(121)  # left side
ax2 = fig.add_subplot(122)  # right side
img = misc.imread('cat.bmp', flatten=True)
print img.shape
result = ndimage.filters.laplace(img,cval=0.0)
ax1.imshow(img)
ax2.imshow(result)
plt.show()


# create laplacian filter
A = np.zeros((7,7))
A[3,3] = 1
print A
B = ndimage.filters.laplace(A)
print B
# convolve
result = ndimage.convolve(img, B, mode='constant', cval=0.0)
fig = plt.figure()
plt.gray()  # show the filtered result in grayscale
plt.imshow(result)
plt.show()

# using gaussian_laplace
# gaussian_laplace(input, sigma, output=None, mode='reflect', cval=0.0)
# sigma is The standard deviations of the Gaussian filter are given for each axis as a sequence, 
# or as a single number, in which case it is equal for all axes..
result = ndimage.filters.gaussian_laplace(img,5,mode='constant')
print result.shape
fig = plt.figure()
plt.gray()  # show the filtered result in grayscale
plt.imshow(result)
plt.show()





