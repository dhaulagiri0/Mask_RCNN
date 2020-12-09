import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema, find_peaks, peak_widths, savgol_filter
from skimage.draw import line
from skimage.morphology import skeletonize
from skimage import data, io, img_as_uint
from skimage.util import invert
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skan import Skeleton, summarize, skeleton_to_csgraph
import cv2
import random
import os
import math
from math import sqrt
from itertools import groupby
from interpolation import *


def getNormal(pt1, pt2, pt3):
  if pt2[1] - pt1[1] != 0: 
    grad1 = (pt2[0] - pt1[0]) / (pt2[1] - pt1[1])
  else: return 0

  if pt3[1] - pt2[1] != 0: 
    grad2 = (pt3[0] - pt2[0]) / (pt3[1] - pt2[1])
  else: return 0

  grad = (grad1 + grad2) / 2

  if grad != 0:
    normalGrad = -1 / grad
  else:
    normalGrad = math.inf

  return normalGrad

def get2Points(x, y, m, halfLen):
  if m == math.inf:
    x1 = x
    x2 = x
    y1 = y + halfLen
    y2 = y - halfLen
  else:
    x1 = x - halfLen/sqrt(m**2 + 1)
    x2 = x + halfLen/sqrt(m**2 + 1)

    y1 = -m*(x - x1) + y
    y2 = -m*(x - x2) + y
  upPt = (y1, x1)
  downPt = (y2, x2)
  return  upPt, downPt

def widthAnalysis(points, im0, n, increment=0.1, show=False, draw_im0 = []):
  widths = []
  coordsList = []
  for i in range(n, len(points) - n):
    coords = points[i]
    coordsNN = points[i + n]
    coordsPP = points[i - n]

    gradient = getNormal(coordsPP, coords, coordsNN)

    # get 2 points each 20 pixels in absolute distance away from the central point
    upPt, downPt = get2Points(coords[1], coords[0], gradient, 20)
    x, y = getPtsAlongLine(downPt, upPt, increment)
    
    # get interpolated pixel values along the line
    pxVal = getInterpolatedPtsVal(y, x, im0)

    # binarize returned values at a threshold of 127.5
    a = np.where(pxVal > 127.5, 1, 0)

    # count the largest consecutive group of activated pixels as the width of the artery
    m = np.r_[False, a==1, False]
    idx = np.flatnonzero(m[:-1]!=m[1:])
    if len(idx > 0):
      largest_island_len = (idx[1::2]-idx[::2]).max()
      totalwidth = largest_island_len * increment
    else:
      # this is the case of total occlusion
      totalwidth = 0

    # draws the checked line if show is enabled
    if show and len(draw_im0) > 0:
      cv2.line(draw_im0, (int(upPt[0]), int(upPt[1])), (int(downPt[0]), int(downPt[1])), (127, 127, 127), thickness=1)
      cv2.imshow('',draw_im0); cv2.waitKey(96)

    widths.append(totalwidth)
    coordsList.append((points[i][0], points[i][1]))
  
  return np.array(widths), coordsList

# get x y coordinates of all points on the skeleton line
def getAllPoints(polys): 
  points = []
  for poly in polys:
    for i in range(0, len(poly) - 1):
      start = poly[i]
      end = poly[i + 1]
      pts = list(zip(*line(*start, *end)))
      points += pts
  return points

def determineStenosisLocation(peak_position, artery_type, artery_length):
  if artery_type == 'lad':
    if peak_position / artery_length > .66:
      # distal
      return 2
    elif peak_position / artery_length > .33:
      # mid
      return 1
    else: 
      # proximal
      return 0
  else:
    if peak_position / artery_length > .55:
      # distal
      return 1
    else:
      # distal
      return 0

def getBoxCoords(centerCoord, length):
  x1 = centerCoord[0] - length
  y1 = centerCoord[1] - length
  x2 = centerCoord[0] + length
  y2 = centerCoord[1] + length
  return {
    'x1' : int(x1),
    'y1' : int(y1),
    'x2' : int(x2),
    'y2' : int(y2)
  }

def scoring(widths, average_width, peaks, artery_type, stenosis_lengths, coordsList, img):
  # widths array records widths from proximal to distal
  # factors are arranged from proximal to distal
  # lcx1 is interpreted as the marginal artery
  # assume right dominant for all cases
  # TODO account for dominance as a user input
  factors_list = {
    'lad' : [3.5, 2.5, 1],
    'diagonal' : [1],
    'lcx1' : [1],
    'lcx2' : [1.5, 1]
  }

  scale = 6. / 10.8
  stenpixels = stenosis_lengths 
  stenosis_lengths = stenosis_lengths * scale

  segments_list = {
    'lad' : ['p', 'm', 'd'],
    'lcx2' : ['p', 'd']
  }

  boxList = []

  factors = factors_list[artery_type]

  artery_length = len(widths)

  score = 1

  stenosis_segments = {}

  # average_width is the average width of the artery taken before smoothing
  # average_width_smoothed = np.average(widths)
  for i in range(0, len(peaks)):
    peak = peaks[i]
    stenosis_length = stenosis_lengths[i]
    width = widths[peak]
    # get Leaman factor for position
    if artery_type == 'lad' or artery_type == 'lcx2':
      location = determineStenosisLocation(peak, artery_type, artery_length)
      segments = segments_list[artery_type]
      segment = segments[location]
      factor = factors[location]
    else:
      factor = factors[0]
      segment = artery_type
      if segment == 'lcx1': segment = 'lcx1'
    
    upperBound = int(peak - stenpixels[i] / 2)
    lowerBound = int(peak + stenpixels[i] / 2)
    if upperBound >= len(widths): upperBound = len(widths) - 1
    if lowerBound >= len(widths): lowerBound = len(widths) - 1
    if lowerBound < 0: lowerBound = 0
    if upperBound < 0: upperBound = 0

    localWidth = (widths[upperBound] + widths[lowerBound]) / 2.

    # sanity check width values
    if round(width, 2) <= 0: width = 0
    if localWidth < 0.5 * average_width: localWidth = average_width

    if width <= average_width:
      percentage = (1. - float(width / average_width)) * 100.
    else:
      percentage = 0

    if percentage < 0:
      percentage = 0

    if segment in stenosis_segments:
      # only score maximum stenosis
      stenosis_segments[segment] = max(stenosis_segments[segment], percentage)
    else:
      stenosis_segments[segment] = percentage

    boxCoords = getBoxCoords(coordsList[peak], int(average_width*1.5))

    if width == 0:
      # occlusion
      # plus one for unknown time of formation
      score += factor * 5 + 1
      c = (255, 0, 0)
      boxList.append(boxCoords)
    elif width < 0.5 * localWidth:
      # significant lesion
      score += factor * 2
      c = (255, 0, 0)
      boxList.append(boxCoords)
    elif width < 0.7 * localWidth:
      c = (128, 0, 128)
      boxList.append(boxCoords)
    elif width < 0.8 * localWidth:
      c = (255, 255, 0)
    else:
      c = (0, 0, 255)

    cv2.rectangle(img, (boxCoords['x1'], boxCoords['y1']), (boxCoords['x2'], boxCoords['y2']), c, 1)
    
    if stenosis_length > 20:
      score += 1
    
  return score, stenosis_segments, boxList

def traverseSkeleton(start, end, graph, coords):
  # gets all the necessary points on the skeleton in the correct sequence
  coordsOrdered = []
  curCoord = start
  rows = np.where(coords == curCoord)
  curIndex = np.argmax(np.bincount(rows[0])) + 1
  prevIndex = -1
  rows = np.where(coords == end)
  endIndex = np.argmax(np.bincount(rows[0]))

  while curIndex != endIndex:
    if prevIndex != -1:
      graph[curIndex][prevIndex] = 0
    coordsOrdered.append(curCoord)
    prevIndex = curIndex
    nonZero = np.nonzero(graph[curIndex])
    if len(nonZero[0]) > 0:
      curIndex = np.nonzero(graph[curIndex])[0][0]
    else:
      break
    curCoord = coords[curIndex - 1]
  coordsOrdered.append(end)
  return np.asarray(coordsOrdered)


def skeletoniseSkimg(maskFilePath):
  image = io.imread(maskFilePath)

  image = rgb2gray(image)

  binary = image > 0

  # perform skeletonization
  skeleton = skeletonize(binary)
  summary = summarize(Skeleton(skeleton, spacing=1))

  # starting coordinate in y, x
  startCoord = [summary.iloc[0]['image-coord-src-0'], summary.iloc[0]['image-coord-src-1']] 
  endCoord = [summary.iloc[0]['image-coord-dst-0'], summary.iloc[0]['image-coord-dst-1']]

  # c0 contains information of all points on the skeleton
  # g0 is the adjacency list 
  g0, c0, _ = skeleton_to_csgraph(skeleton, spacing=1)
  all_pts = c0[1:]

  # get points along skeleton in the correct sequence
  pts = traverseSkeleton(startCoord, endCoord, g0.toarray(), all_pts)

  return pts.astype(int)

def getScore(filename, folderDirectory='A:/segmented/', show=False, save=False):

  pathPrefix = f"{folderDirectory}/{filename.split('_')[0]}/{filename.split('.')[0].split('_')[0]}_{filename.split('.')[0].split('_')[1]}"
  
  artery_type = filename.split('_')[-1]

  # default should be A:/segmented/
  all_pts = skeletoniseSkimg(f"{pathPrefix}/{filename}_bin_mask.png")
  all_pts = np.flip(all_pts, 1)

  imSegmented = cv2.imread(f'{pathPrefix}/{filename}_segmented_threshold_binary.png', 0)
  imDisplay = cv2.imread(f"{pathPrefix}/{filename.split('_')[0] + '_' + filename.split('_')[1]}_original.png")
  # print(f"{pathPrefix}/{filename.split('_')[0] + '_' + filename.split('_')[1]}_original.png")

  widths, coordsList = widthAnalysis(all_pts, imSegmented, 4, show=False, draw_im0=imSegmented.copy())

  window_size = 21

  # sanity check in case there are multiple skeletons
  if len(widths) > window_size:
    # smoothing width graph
    widths_s = savgol_filter(widths, window_size, 3)
  else:
    return None, None, None

  segment_widths = []
  segment_offsets = []
  # split widths into segments
  if artery_type == 'lcx2' or artery_type == 'lad':
    segment_widths.append(widths_s[:int(len(widths_s)/3)])
    segment_widths.append(widths_s[int(len(widths_s)/3):int(len(widths_s) * 2/3)])
    segment_widths.append(widths_s[int(len(widths_s) * 2/3):])

    segment_offsets = [0,  int(len(widths)/3), int(len(widths) * 2/3)]
  else:
    segment_widths.append(widths_s[:int(len(widths)/2)])
    segment_widths.append(widths_s[int(len(widths)/2):])

    segment_offsets = [0,  int(len(widths)/2)]

  average_widths = []
  for segment in segment_widths:
    average_widths.append(np.average(segment)) 

  # locate dips in widths and identify them as regions of stenosis
  l = []
  for i in range(len(segment_widths)):
    segment = segment_widths[i]
    average_width = average_widths[i]
    peaks, _ = find_peaks(np.negative(segment), distance=5, prominence=(average_width*0.15, None), width=(1, None))
    peaks = np.array(peaks) + segment_offsets[i]
    peaks = peaks.tolist()
    l += peaks
  peaks = l

  # determine length of each detected stenosis
  stenosis_lengths_ = peak_widths(np.negative(widths_s), peaks, rel_height=0.7)
  stenosis_lengths = stenosis_lengths_[0]

  # get syntax score and highest stenosis percentages for each segment of the artery if any
  # box coords are collated to calculate f1 score later
  score, percentages, boxList = scoring(widths_s, np.array(average_widths).mean(), peaks, artery_type, stenosis_lengths, coordsList, imDisplay)

  # plotting for display purposes
  plt.cla()
  plt.plot(range(1, len(widths_s) + 1), widths_s)
  plt.plot(peaks, widths_s[peaks], "x")
  
  # draw lines representing the length of stenosis
  # diabled becaue it causes the graph to be compressed 
  # (the lines are always negative due to the way it is generated)
  #plt.hlines(*stenosis_lengths_[1:], color="C2")

  # displays plot and stenosis locations
  # if activated at the same time as save, plots go blank
  if show:
    cv2.imshow('stenosis locations', imDisplay); cv2.waitKey(0)
    plt.show()

  # saves all the plots and stenosis location diagram into specified directory
  if save:
    if not os.path.exists(f"{folderDirectory.split('/')[0]}/detections"):
      os.mkdir(f"{folderDirectory.split('/')[0]}/detections")

    outPath = f"{folderDirectory.split('/')[0]}/detections/{filename}"
    cv2.imwrite(outPath + '_stenosis_locations.png', imDisplay)
    plt.savefig(outPath + '_width_plot.png')

  return score, percentages, boxList

# score, percentages, bboxList = getScore('1738_47_lcx1', folderDirectory='A:/segmented_manual/', show=True, save=False)
# print(score)
# print(percentages)

# legacy code
# if __name__ == "__main__":
#   import cv2
#   import random

#   filename = '2477_25_lad'
#   all_pts = skeletoniseSkimg(f'A:/segmented/{filename}bin_mask.png')
#   all_pts = np.flip(all_pts, 1)
#   # all_pts = locateAllPoints(destpt, skeleton)
#   im0 = cv2.imread(f'A:/segmented/{filename}bin_mask.png')
#   imSegmented = cv2.imread(f'A:/segmented/{filename}segmented_threshold_binary.png')

#   im = (im0[:,:,0]>128).astype(np.uint8)

#   # # for i in range(im.shape[0]):
#   # #   for j in range(im.shape[1]):
#   # #     print(im[i,j],end="")
#   # #   print("")
#   # # print(np.sum(im),im.shape[0]*im.shape[1])
#   # im = thinning(im);


#   # rects = []
  
#   # polys = traceSkeleton(im,0,0,im.shape[1],im.shape[0],10,999,rects)
  
#   # points = getAllPoints(polys)
#   # print(points)

#   arr, coordsList = widthAnalysis(all_pts, imSegmented, 1)
#   arr = np.array(arr)
#   arr_s = smooth(arr, 22)
#   average_width = np.average(arr) 

#   peaks, properties = find_peaks(np.negative(arr_s), distance=5, prominence=(average_width*0.15, None), width=(1, None))
  
#   # plt.plot(range(1, len(arr) + 1), arr)
#   plt.plot(range(1, len(arr_s) + 1), arr_s)
#   plt.plot(peaks, arr_s[peaks], "x")
#   plt.show()
  
#   circleIm = imSegmented
#   for peak in peaks:
#     # print(int((peak / len(peaks)) * len(coordsList)))
#     c = (200*random.random(),200*random.random(),200*random.random())
#     cv2.circle(circleIm, coordsList[int((peak / len(arr_s)) * len(coordsList))], 2, c, 2)

#   # cv2.imshow('',circleIm);cv2.waitKey(0)
#   cv2.imshow('',imSegmented);cv2.waitKey(0)

#   # for l in polys:
#   #     c = (200*random.random(),200*random.random(),200*random.random())
#   #     for i in range(0,len(l)-1):
#   #       cv2.line(imSegmented,(l[i][0],l[i][1]),(l[i+1][0],l[i+1][1]), c, thickness=2)

#   # cv2.imshow('',imSegmented);cv2.waitKey(0)

#   score = scoring(arr_s, average_width, peaks, filename.split('_')[-1])
#   print(f'Computed Syntax score for this artery: {score}')



# # binary image thinning (skeletonization) in-place.
# # implements Zhang-Suen algorithm.
# # http://agcggs680.pbworks.com/f/Zhan-Suen_algorithm.pdf
# # @param im   the binary image
# def thinningZS(im):
#   prev = np.zeros(im.shape,np.uint8);
#   while True:
#     im = thinningZSIteration(im,0);
#     im = thinningZSIteration(im,1)
#     diff = np.sum(np.abs(prev-im));
#     if not diff:
#       break
#     prev = im
#   return im

# # 1 pass of Zhang-Suen thinning 
# def thinningZSIteration(im, iter):
#   marker = np.zeros(im.shape,np.uint8);
#   for i in range(1,im.shape[0]-1):
#     for j in range(1,im.shape[1]-1):
#       p2 = im[(i-1),j]  ;
#       p3 = im[(i-1),j+1];
#       p4 = im[(i),j+1]  ;
#       p5 = im[(i+1),j+1];
#       p6 = im[(i+1),j]  ;
#       p7 = im[(i+1),j-1];
#       p8 = im[(i),j-1]  ;
#       p9 = im[(i-1),j-1];
#       A  = (p2 == 0 and p3) + (p3 == 0 and p4) + \
#            (p4 == 0 and p5) + (p5 == 0 and p6) + \
#            (p6 == 0 and p7) + (p7 == 0 and p8) + \
#            (p8 == 0 and p9) + (p9 == 0 and p2);
#       B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
#       m1 = (p2 * p4 * p6) if (iter == 0 ) else (p2 * p4 * p8);
#       m2 = (p4 * p6 * p8) if (iter == 0 ) else (p2 * p6 * p8);

#       if (A == 1 and (B >= 2 and B <= 6) and m1 == 0 and m2 == 0):
#         marker[i,j] = 1;

#   return np.bitwise_and(im,np.bitwise_not(marker))


# def thinningSkimage(im):
#   from skimage.morphology import skeletonize
#   return skeletonize(im).astype(np.uint8)

# def thinning(im):
#   try:
#     return thinningSkimage(im)
#   except:
#     return thinningZS(im)

# #check if a region has any white pixel
# def notEmpty(im, x, y, w, h):
#   return np.sum(im) > 0


# # merge ith fragment of second chunk to first chunk
# # @param c0   fragments from first  chunk
# # @param c1   fragments from second chunk
# # @param i    index of the fragment in first chunk
# # @param sx   (x or y) coordinate of the seam
# # @param isv  is vertical, not horizontal?
# # @param mode 2-bit flag, 
# #             MSB = is matching the left (not right) end of the fragment from first  chunk
# #             LSB = is matching the right (not left) end of the fragment from second chunk
# # @return     matching successful?             
# # 
# def mergeImpl(c0, c1, i, sx, isv, mode):

#   B0 = (mode >> 1 & 1)>0; # match c0 left
#   B1 = (mode >> 0 & 1)>0; # match c1 left
#   mj = -1;
#   md = 4; # maximum offset to be regarded as continuous
  
#   p1 = c1[i][0 if B1 else -1];
  
#   if (abs(p1[isv]-sx)>0): # not on the seam, skip
#     return False
  
#   # find the best match
#   for j in range(len(c0)):
#     p0 = c0[j][0 if B0 else -1];
#     if (abs(p0[isv]-sx)>1): # not on the seam, skip
#       continue
    
#     d = abs(p0[not isv] - p1[not isv]);
#     if (d < md):
#       mj = j;
#       md = d;

#   if (mj != -1): # best match is good enough, merge them
#     if (B0 and B1):
#       c0[mj] = list(reversed(c1[i])) + c0[mj]
#     elif (not B0 and B1):
#       c0[mj]+=c1[i]
#     elif (B0 and not B1):
#       c0[mj] = c1[i] + c0[mj]
#     else:
#       c0[mj] += list(reversed(c1[i]))
    
#     c1.pop(i);
#     return True;
#   return False;

# HORIZONTAL = 1;
# VERTICAL = 2;

# # merge fragments from two chunks
# # @param c0   fragments from first  chunk
# # @param c1   fragments from second chunk
# # @param sx   (x or y) coordinate of the seam
# # @param dr   merge direction, HORIZONTAL or VERTICAL?
# # 
# def mergeFrags(c0, c1, sx, dr):
#   for i in range(len(c1)-1,-1,-1):
#     if (dr == HORIZONTAL):
#       if (mergeImpl(c0,c1,i,sx,False,1)):continue;
#       if (mergeImpl(c0,c1,i,sx,False,3)):continue;
#       if (mergeImpl(c0,c1,i,sx,False,0)):continue;
#       if (mergeImpl(c0,c1,i,sx,False,2)):continue;
#     else:
#       if (mergeImpl(c0,c1,i,sx,True,1)):continue;
#       if (mergeImpl(c0,c1,i,sx,True,3)):continue;
#       if (mergeImpl(c0,c1,i,sx,True,0)):continue;
#       if (mergeImpl(c0,c1,i,sx,True,2)):continue;      
    
#   c0 += c1


# # recursive bottom: turn chunk into polyline fragments;
# # look around on 4 edges of the chunk, and identify the "outgoing" pixels;
# # add segments connecting these pixels to center of chunk;
# # apply heuristics to adjust center of chunk
# # 
# # @param im   the bitmap image
# # @param x    left of   chunk
# # @param y    top of    chunk
# # @param w    width of  chunk
# # @param h    height of chunk
# # @return     the polyline fragments
# # 
# def chunkToFrags(im, x, y, w, h):
#   frags = []
#   on = False; # to deal with strokes thicker than 1px
#   li=-1; lj=-1;
  
#   # walk around the edge clockwise
#   for k in range(h+h+w+w-4):
#     i=0; j=0;
#     if (k < w):
#       i = y+0; j = x+k;
#     elif (k < w+h-1):
#       i = y+k-w+1; j = x+w-1;
#     elif (k < w+h+w-2):
#       i = y+h-1; j = x+w-(k-w-h+3); 
#     else:
#       i = y+h-(k-w-h-w+4); j = x+0;
    
#     if (im[i,j]): # found an outgoing pixel
#       if (not on):     # left side of stroke
#         on = True;
#         frags.append([[j,i],[x+w//2,y+h//2]])
#     else:
#       if (on):# right side of stroke, average to get center of stroke
#         frags[-1][0][0]= (frags[-1][0][0]+lj)//2;
#         frags[-1][0][1]= (frags[-1][0][1]+li)//2;
#         on = False;
#     li = i;
#     lj = j;
  
#   if (len(frags) == 2): # probably just a line, connect them
#     f = [frags[0][0],frags[1][0]];
#     frags.pop(0);
#     frags.pop(0);
#     frags.append(f);
#   elif (len(frags) > 2): # it's a crossroad, guess the intersection
#     ms = 0;
#     mi = -1;
#     mj = -1;
#     # use convolution to find brightest blob
#     for i in range(y+1,y+h-1):
#       for j in range(x+1,x+w-1):
#         s = \
#           (im[i-1,j-1]) + (im[i-1,j]) +(im[i-1,j+1])+\
#           (im[i,j-1]  ) +   (im[i,j]) +    (im[i,j+1])+\
#           (im[i+1,j-1]) + (im[i+1,j]) +  (im[i+1,j+1]);
#         if (s > ms):
#           mi = i;
#           mj = j;
#           ms = s;
#         elif (s == ms and abs(j-(x+w//2))+abs(i-(y+h//2)) < abs(mj-(x+w//2))+abs(mi-(y+h//2))):
#           mi = i;
#           mj = j;
#           ms = s;

#     if (mi != -1):
#       for i in range(len(frags)):
#         frags[i][1]=[mj,mi]
#   return frags;


# # Trace skeleton from thinning result.
# # Algorithm:
# # 1. if chunk size is small enough, reach recursive bottom and turn it into segments
# # 2. attempt to split the chunk into 2 smaller chunks, either horizontall or vertically;
# #    find the best "seam" to carve along, and avoid possible degenerate cases
# # 3. recurse on each chunk, and merge their segments
# # 
# # @param im      the bitmap image
# # @param x       left of   chunk
# # @param y       top of    chunk
# # @param w       width of  chunk
# # @param h       height of chunk
# # @param csize   chunk size
# # @param maxIter maximum number of iterations
# # @param rects   if not null, will be populated with chunk bounding boxes (e.g. for visualization)
# # @return        an array of polylines
# # 
# def traceSkeleton(im, x, y, w, h, csize, maxIter, rects):
  
#   frags = []
  
#   if (maxIter == 0): # gameover
#     return frags;
#   if (w <= csize and h <= csize): # recursive bottom
#     frags += chunkToFrags(im,x,y,w,h);
#     return frags;
  
#   ms = im.shape[0]+im.shape[1]; # number of white pixels on the seam, less the better
#   mi = -1; # horizontal seam candidate
#   mj = -1; # vertical   seam candidate
  
#   if (h > csize): # try splitting top and bottom
#     for i in range(y+3,y+h-3):
#       if (im[i,x]  or im[(i-1),x]  or im[i,x+w-1]  or im[(i-1),x+w-1]):
#         continue
      
#       s = 0;
#       for j in range(x,x+w):
#         s += im[i,j];
#         s += im[(i-1),j];
      
#       if (s < ms):
#         ms = s; mi = i;
#       elif (s == ms  and  abs(i-(y+h//2))<abs(mi-(y+h//2))):
#         # if there is a draw (very common), we want the seam to be near the middle
#         # to balance the divide and conquer tree
#         ms = s; mi = i;
  
#   if (w > csize): # same as above, try splitting left and right
#     for j in range(x+3,x+w-2):
#       if (im[y,j] or im[(y+h-1),j] or im[y,j-1] or im[(y+h-1),j-1]):
#         continue
      
#       s = 0;
#       for i in range(y,y+h):
#         s += im[i,j];
#         s += im[i,j-1];
#       if (s < ms):
#         ms = s;
#         mi = -1; # horizontal seam is defeated
#         mj = j;
#       elif (s == ms  and  abs(j-(x+w//2))<abs(mj-(x+w//2))):
#         ms = s;
#         mi = -1;
#         mj = j;

#   nf = []; # new fragments
#   if (h > csize  and  mi != -1): # split top and bottom
#     L = [x,y,w,mi-y];    # new chunk bounding boxes
#     R = [x,mi,w,y+h-mi];
    
#     if (notEmpty(im,L[0],L[1],L[2],L[3])): # if there are no white pixels, don't waste time
#       if(rects!=None):rects.append(L);
#       nf += traceSkeleton(im,L[0],L[1],L[2],L[3],csize,maxIter-1,rects) # recurse
    
#     if (notEmpty(im,R[0],R[1],R[2],R[3])):
#       if(rects!=None):rects.append(R);
#       mergeFrags(nf,traceSkeleton(im,R[0],R[1],R[2],R[3],csize,maxIter-1,rects),mi,VERTICAL);
    
#   elif (w > csize  and  mj != -1): # split left and right
#     L = [x,y,mj-x,h];
#     R = [mj,y,x+w-mj,h];
#     if (notEmpty(im,L[0],L[1],L[2],L[3])):
#       if(rects!=None):rects.append(L);
#       nf+=traceSkeleton(im,L[0],L[1],L[2],L[3],csize,maxIter-1,rects);
    
#     if (notEmpty(im,R[0],R[1],R[2],R[3])):
#       if(rects!=None):rects.append(R);
#       mergeFrags(nf,traceSkeleton(im,R[0],R[1],R[2],R[3],csize,maxIter-1,rects),mj,HORIZONTAL);
    
#   frags+=nf;
#   if (mi == -1  and  mj == -1): # splitting failed! do the recursive bottom instead
#     frags += chunkToFrags(im,x,y,w,h);
  
#   return frags

# def smooth(x,window_len=11,window='hanning'):
#     """smooth the data using a window with requested size.
    
#     This method is based on the convolution of a scaled window with the signal.
#     The signal is prepared by introducing reflected copies of the signal 
#     (with the window size) in both ends so that transient parts are minimized
#     in the begining and end part of the output signal.
    
#     input:
#         x: the input signal 
#         window_len: the dimension of the smoothing window; should be an odd integer
#         window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
#             flat window will produce a moving average smoothing.

#     output:
#         the smoothed signal
        
#     example:

#     t=linspace(-2,2,0.1)
#     x=sin(t)+randn(len(t))*0.1
#     y=smooth(x)
    
#     see also: 
    
#     numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
#     scipy.signal.lfilter
 
#     NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
#     """

#     if x.ndim != 1:
#         raise ValueError("smooth only accepts 1 dimension arrays.")

#     if x.size < window_len:
#         raise ValueError("Input vector needs to be bigger than window size.")


#     if window_len<3:
#         return x


#     if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
#         raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


#     s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
#     #print(len(s))
#     if window == 'flat': #moving average
#         w=np.ones(window_len,'d')
#     else:
#         w=eval('np.'+window+'(window_len)')

#     y=np.convolve(w/w.sum(),s,mode='same')
#     return y

# def locateAllPoints(destpt, skeleton):
#   pts = []
#   pts.append(destpt)

#   element = skeleton[destpt[1]][destpt[0]]
#   nextPoint = True
#   while nextPoint:
#     for n in range(destpt[1] - 1, destpt[1] + 1):
#       if n < 0 or n > skeleton.shape[0]: continue
#       for m in range(destpt[0] - 1, destpt[0] + 1):
#         if m < 0 or m > skeleton.shape[1]: continue
#         if (m, n) not in pts and skeleton[n, m]:
#           pts.append((m, n))

#   return pts

# def widthAnalysisLegacy(points, im0, n, increment=0.1, show=False, draw_im0 = []):
#   widths = []
#   coordsList = []
#   for i in range(n, len(points) - n):
#     coords = points[i]
#     coordsNN = points[i + n]
#     coordsPP = points[i - n]

#     gradient = getNormal(coordsPP, coords, coordsNN)

#     # # check up and down width 
#     # upwidth = 0
#     # downwidth = 0

#     # curcoord = coords
#     # cnt = 0

#     # # check up width
#     # while 0 <= curcoord[1] < im0.shape[1] and 0 <= curcoord[0] < im0.shape[0] and (im0[curcoord[1]][curcoord[0]].any() or cnt <= 3):
#     #   cnt += 1
#     #   if not math.isnan(gradient) and not math.isinf(gradient):
#     #     # print(gradient)
#     #     x2 = curcoord[1] + 1
#     #     y2 = int(gradient * x2 - coords[1] * gradient + coords[0])
#     #   else:
#     #     x2 = curcoord[1]
#     #     y2 = curcoord[0] + 1

#     #   curcoord = [y2, x2]
    
#     # upcoords = curcoord
#     # curcoord = coords

#     # cnt = 0

#     # # check down width
#     # while 0 <= curcoord[1] < im0.shape[1] and 0 <= curcoord[0] < im0.shape[0] and (im0[curcoord[1]][curcoord[0]].any() or cnt <= 3):
#     #   cnt += 1
#     #   if not math.isnan(gradient) and not math.isinf(gradient):
#     #     x2 = curcoord[1] - 1
#     #     y2 = int(gradient * x2 - coords[1] * gradient + coords[0])
#     #   else:
#     #     x2 = curcoord[1]
#     #     y2 = curcoord[0] - 1

#     #   # downwidth += math.sqrt((y2 - curcoord[0])**2 + (x2-curcoord[1])**2)
#     #   curcoord = [y2, x2]

#     # downcoords = curcoord
#     # upPt = upcoords
#     # downPt = downcoords
#     upPt, downPt = get2Points(coords[1], coords[0], gradient, 20)
#     x, y = getPtsAlongLine(downPt, upPt, increment)
#     pxVal = getInterpolatedPtsVal(y, x, im0)

#     # totalwidth = math.sqrt((upcoords[0] - downcoords[0])**2 + (upcoords[1] - downcoords[1])**2)
#     # totalwidth1 = (pxVal > 127.5).sum() * increment
#     a = np.where(pxVal > 127.5, 1, 0)

#     m = np.r_[False, a==1, False]
#     idx = np.flatnonzero(m[:-1]!=m[1:])
#     if len(idx > 0):
#       largest_island_len = (idx[1::2]-idx[::2]).max()
#       totalwidth = largest_island_len * increment
#     else:
#       totalwidth = 0

#     if show and len(draw_im0) > 0:
#       cv2.line(draw_im0, (int(upPt[0]), int(upPt[1])), (int(downPt[0]), int(downPt[1])), (127, 127, 127), thickness=1)
#       cv2.imshow('',draw_im0);cv2.waitKey(96)

#     widths.append(totalwidth)
#     coordsList.append((points[i][0], points[i][1]))

#     # arr.append([coords[1], coords[0], gradient, upwidth, downwidth, totalwidth]) 
  
#   return widths, coordsList
