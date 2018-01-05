# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 09:48:50 2015

@author: bxiao
"""

import numpy as np
import matplotlib.pyplot as plt


from skimage.transform import pyramid_gaussian
from scipy import misc

from skimage import data

#image = data.astronaut()
image = misc.lena()
image = np.float64(image)

pyramid = tuple(pyramid_gaussian(image, downscale=2))


rows, cols = image.shape
composite_image = np.zeros((rows, cols + cols / 2), dtype=np.double)

composite_image[:rows, :cols] = pyramid[0]

i_row = 0
for p in pyramid[1:]:
    n_rows, n_cols = p.shape[:2]
    composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
    i_row += n_rows


print i_row
fig, ax = plt.subplots()
ax.imshow(composite_image,cmap='gray')
plt.show()
