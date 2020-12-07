import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2
from decimal import Decimal
import math
from math import sqrt

# gets a point thats a certain unit (whatever value increment is)
# away from the first the coordinates (x, y) along the given line
# the point given will have a smaller x value than the given x
# x, y should always be a point on the right of the next point on the x-axis
def getNextPoint(x, y, m, increment):
  if m == math.inf:
    x1 = x
    y1 = y + increment
  else:
    x1 = x - increment/sqrt(m**2 + 1)
    y1 = -m*(x - x1) + y
  return  x1, y1

# given two points, find points that are along the line
# each point is one increment in absolute distance away from the previous point along the line
# the sequence of the points given does not matter
def getPtsAlongLine(srcPt, dstPt, increment=0.1):
    x1, y1 = srcPt
    x2, y2 = dstPt
    xcoords = []
    ycoords = []
    # find out which point is on the right along the x-axis
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
            # get next point to the left of the current one along the line
            x, y = getNextPoint(xcoords[-1], ycoords[-1], gradient, increment)
            xcoords.append(x)
            ycoords.append(y)
    else: 
        # its a straight line parallel to the y-axis
        ycoords = np.arange(min(y1, y2), max(y1, y2) + increment, increment)
        xcoords = [x1] * len(ycoords)

    return np.asarray(xcoords), np.asarray(ycoords)


# x and y are lists of coordinates along the normal line
# returns all the interpolated pixel values of the image along the given line
def getInterpolatedPtsVal(x, y, img):
    xmax = img.shape[0]
    ymax = img.shape[1]
    xcoords = np.arange(xmax)
    ycoords = np.arange(ymax)
    sp = RectBivariateSpline(xcoords, ycoords, img)
    return sp(x, y, grid=False)


# legacy code 
# def getPtsAlongLineOld(srcPt, dstPt, increment=0.1):
#     x1, y1 = srcPt
#     x2, y2 = dstPt
#     if x2 - x1 != 0:
#         gradient = (y2 - y1) / (x2 - x1)
#         xcoords = np.arange(min(x1, x2), max(x1, x2) + increment, increment)
#         ycoords = []
#         for x in xcoords:
#             y = gradient*(x - x1) + y1
#             ycoords.append(y)
#     else: 
#         ycoords = np.arange(min(y1, y2), max(y1, y2) + increment, increment)
#         xcoords = [x1] * len(ycoords)

#     return np.asarray(xcoords), np.asarray(ycoords)

