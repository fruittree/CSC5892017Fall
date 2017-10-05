import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack

random_img = np.random.random((256,256))

Z = np.random.random((256,256))   # Test data

# compute the FFT
f = np.fft.fft2(random_img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 30*np.log(np.abs(fshift))

# display the original image and FFT
plt.subplot(121),plt.imshow(random_img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Log Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

# diagonal image
diag_img = 100*np.identity(256)


# compute the FFT
f = np.fft.fft2(diag_img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 10*np.log(np.abs(fshift))

# display the original image and FFT
plt.subplot(121),plt.imshow(diag_img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Log Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()




