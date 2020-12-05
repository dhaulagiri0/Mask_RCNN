import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2
from decimal import Decimal
import math
from math import sqrt

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

def getPtsAlongLineOld(srcPt, dstPt, increment=0.1):
    x1, y1 = srcPt
    x2, y2 = dstPt
    if x2 - x1 != 0:
        gradient = (y2 - y1) / (x2 - x1)
        xcoords = np.arange(min(x1, x2), max(x1, x2) + increment, increment)
        ycoords = []
        for x in xcoords:
            y = gradient*(x - x1) + y1
            ycoords.append(y)
    else: 
        ycoords = np.arange(min(y1, y2), max(y1, y2) + increment, increment)
        xcoords = [x1] * len(ycoords)

    return np.asarray(xcoords), np.asarray(ycoords)

def getNextPoint(x, y, m, increment):
  if m == math.inf:
    x1 = x
    y1 = y + increment
  else:
    x1 = x - increment/sqrt(m**2 + 1)
    y1 = -m*(x - x1) + y
  return  x1, y1

def getPtsAlongLine(srcPt, dstPt, increment=0.1):
    x1, y1 = srcPt
    x2, y2 = dstPt
    xcoords = []
    ycoords = []
    if x1 > x2:
        xcoords.append(x1)
        ycoords.append(y1)
    else:
        xcoords.append(x2)
        ycoords.append(y2)
    if x2 - x1 != 0:
        gradient = (y2 - y1) / (x2 - x1)
        distance = sqrt((x1 + x2)**2 + (y1 + y2)**2)
        steps = int(distance / increment)
        for _ in range(steps):
            x, y = getNextPoint(xcoords[-1], ycoords[-1], gradient, increment)
            xcoords.append(x)
            ycoords.append(y)
        # xcoords = np.arange(min(x1, x2), max(x1, x2) + increment, increment)
        # ycoords = []
        # for x in xcoords:
        #     y = gradient*(x - x1) + y1
        #     ycoords.append(y)
    else: 
        ycoords = np.arange(min(y1, y2), max(y1, y2) + increment, increment)
        xcoords = [x1] * len(ycoords)

    return np.asarray(xcoords), np.asarray(ycoords)


# x and y are lists of coordinates along the normal line
def getInterpolatedPtsVal(x, y, img):
    xmax = img.shape[0]
    ymax = img.shape[1]
    xcoords = np.arange(xmax)
    ycoords = np.arange(ymax)
    sp = RectBivariateSpline(xcoords, ycoords, img)
    return sp(x, y, grid=False)

# img = cv2.imread('A:/segmented/1367/1367_35/1367_35_lcx1segmented_threshold_binary.png', 0)
# x, y = getPtsAlongLine((248, 118), (256, 109))
# pxVal = getInterpolatedPtsVal(y, x, img)
# print(np.amax(img))
# # # sp = getInterpolatedImg(img)
# # # pxVal = getPtsAlongLine((195, 272), (185, 282), -1, sp)
# print(x, y)
# print(pxVal)
# print((pxVal > 127.5).sum())