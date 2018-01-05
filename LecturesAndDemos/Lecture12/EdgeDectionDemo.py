import numpy as np
from numpy import pi
import scipy.ndimage as ndi

def non_maximal_edge_suppresion(mag, orient):
    """Non Maximal suppression of gradient magnitude and orientation."""
    # bin orientations into 4 discrete directions
    abin = ((orient + pi) * 4 / pi + 0.5).astype('int') % 4
    mask = np.zeros(mag.shape, dtype='bool')
    mask[1:-1,1:-1] = True
    edge_map = np.zeros(mag.shape, dtype='bool')
    offsets = ((1,0), (1,1), (0,1), (-1,1))    
    for a, (di, dj) in zip(range(4), offsets):
        cand_idx = np.nonzero(np.logical_and(abin==a, mask))
        for i,j in zip(*cand_idx):
            if mag[i,j] > mag[i+di,j+dj] and mag[i,j] > mag[i-di,j-dj]:
                edge_map[i,j] = True
    return edge_map

# here finisht he canny_edge detetio algorithm using the above function. 
def canny_edges(image, sigma=1.0, low_thresh=50, high_thresh=100):
     """Compute Canny edge detection on an image."""

    # step 1 Gaussian filtering, using a gaussian filter with xigma = 1. 
    image = ndi.filters.gaussian_filter(image, sigma)
    dx = ndi.filters.sobel(image,0)
    dy = ndi.filters.sobel(image,1)


    # step 2 computing gradient magnitude and orientation 
    mag = np.hypot(dx,dy)
    ort = np.arctan2(dy, dx)

    # step 3: perfomring non-maximum suppression upper threshold
    edge_map = non_maximal_edge_suppresion(mag,ort)
     

    # step 4: threading 


    # labeling edge maps
    # this is doing hystersis threshold
    #labels, num_labels = ndi.measurements.label(edge_map, np.ones((3,3)))

    return edge_map


if __name__ == "__main__":
    import scipy.misc
    ascent = scipy.misc.ascent()
    from matplotlib.pyplot import imshow, gray, show
    imshow(canny_edges(ascent, 1))
    gray()
    show()