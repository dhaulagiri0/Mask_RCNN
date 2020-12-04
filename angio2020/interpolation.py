import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2
from decimal import Decimal

# xmax = 512
# ymax = 512

# xcoords = np.arange(xmax)
# ycoords = np.arange(ymax)

# sp = RectBivariateSpline(xcoords, ycoords, image)
# # dx2, dy2 = 0.16, 0.16
# # x2 = np.arange(xmin, xmax, dx2)
# # y2 = np.arange(ymin, ymax, dy2)
# #y value first then x
# Z2 = sp(266, 286, grid=False)
# Z3 = sp(290.1, 230.1, grid=False)
# print(round(Z2[()], 10))
# # cv2.imshow('',Z2);cv2.waitKey(0)

def getPtsAlongLine(srcPt, dstPt, gradient, increment=0.1):
    y1, x1 = srcPt
    y2, x2 = dstPt
    xcoords = np.arange(x1, x2 + increment, 0.1)
    ycoords = []
    for x in xcoords:
        y = gradient*(x - x1) + y1
        ycoords.append(round(y, 1))
    return np.asarray(xcoords), np.asarray(ycoords)


# x and y are lists of coordinates along the normal line
def getInterpolatedPtsVal(x, y, img):
    xmax = img.shape[0]
    ymax = img.shape[1]
    xcoords = np.arange(xmax)
    ycoords = np.arange(ymax)
    sp = RectBivariateSpline(xcoords, ycoords, img)
    return sp(y, x, grid=False)

img = cv2.imread('A:/segmented/1367/1367_35/1367_35_diagonalbin_mask.png', 0)
x, y = getPtsAlongLine((195, 272), (185, 282), -1)
pxVal = getInterpolatedPtsVal(x, y, img)

# sp = getInterpolatedImg(img)
# pxVal = getPtsAlongLine((195, 272), (185, 282), -1, sp)
print(pxVal)
print((pxVal > 0.1).sum())