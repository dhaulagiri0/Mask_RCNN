import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2

image = cv2.imread('B:/segmented/1367/1367_35/1367_35_diagonalbin_mask.png', 0)

xcoords = np.arange(512)
ycoords = np.arange(512)

sp = RectBivariateSpline(xcoords, ycoords, image)

print(sp)