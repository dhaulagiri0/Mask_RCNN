# trace_skeleton.py
# Trace skeletonization result into polylines
#
# Lingdong Huang 2020

import numpy as np
import cv2
from skimage.draw import line
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema, find_peaks
import math

# binary image thinning (skeletonization) in-place.
# implements Zhang-Suen algorithm.
# http://agcggs680.pbworks.com/f/Zhan-Suen_algorithm.pdf
# @param im   the binary image
def thinningZS(im):
  prev = np.zeros(im.shape,np.uint8);
  while True:
    im = thinningZSIteration(im,0);
    im = thinningZSIteration(im,1)
    diff = np.sum(np.abs(prev-im));
    if not diff:
      break
    prev = im
  return im

# 1 pass of Zhang-Suen thinning 
def thinningZSIteration(im, iter):
  marker = np.zeros(im.shape,np.uint8);
  for i in range(1,im.shape[0]-1):
    for j in range(1,im.shape[1]-1):
      p2 = im[(i-1),j]  ;
      p3 = im[(i-1),j+1];
      p4 = im[(i),j+1]  ;
      p5 = im[(i+1),j+1];
      p6 = im[(i+1),j]  ;
      p7 = im[(i+1),j-1];
      p8 = im[(i),j-1]  ;
      p9 = im[(i-1),j-1];
      A  = (p2 == 0 and p3) + (p3 == 0 and p4) + \
           (p4 == 0 and p5) + (p5 == 0 and p6) + \
           (p6 == 0 and p7) + (p7 == 0 and p8) + \
           (p8 == 0 and p9) + (p9 == 0 and p2);
      B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
      m1 = (p2 * p4 * p6) if (iter == 0 ) else (p2 * p4 * p8);
      m2 = (p4 * p6 * p8) if (iter == 0 ) else (p2 * p6 * p8);

      if (A == 1 and (B >= 2 and B <= 6) and m1 == 0 and m2 == 0):
        marker[i,j] = 1;

  return np.bitwise_and(im,np.bitwise_not(marker))


def thinningSkimage(im):
  from skimage.morphology import skeletonize
  return skeletonize(im).astype(np.uint8)

def thinning(im):
  try:
    return thinningSkimage(im)
  except:
    return thinningZS(im)

#check if a region has any white pixel
def notEmpty(im, x, y, w, h):
  return np.sum(im) > 0


# merge ith fragment of second chunk to first chunk
# @param c0   fragments from first  chunk
# @param c1   fragments from second chunk
# @param i    index of the fragment in first chunk
# @param sx   (x or y) coordinate of the seam
# @param isv  is vertical, not horizontal?
# @param mode 2-bit flag, 
#             MSB = is matching the left (not right) end of the fragment from first  chunk
#             LSB = is matching the right (not left) end of the fragment from second chunk
# @return     matching successful?             
# 
def mergeImpl(c0, c1, i, sx, isv, mode):

  B0 = (mode >> 1 & 1)>0; # match c0 left
  B1 = (mode >> 0 & 1)>0; # match c1 left
  mj = -1;
  md = 4; # maximum offset to be regarded as continuous
  
  p1 = c1[i][0 if B1 else -1];
  
  if (abs(p1[isv]-sx)>0): # not on the seam, skip
    return False
  
  # find the best match
  for j in range(len(c0)):
    p0 = c0[j][0 if B0 else -1];
    if (abs(p0[isv]-sx)>1): # not on the seam, skip
      continue
    
    d = abs(p0[not isv] - p1[not isv]);
    if (d < md):
      mj = j;
      md = d;

  if (mj != -1): # best match is good enough, merge them
    if (B0 and B1):
      c0[mj] = list(reversed(c1[i])) + c0[mj]
    elif (not B0 and B1):
      c0[mj]+=c1[i]
    elif (B0 and not B1):
      c0[mj] = c1[i] + c0[mj]
    else:
      c0[mj] += list(reversed(c1[i]))
    
    c1.pop(i);
    return True;
  return False;

HORIZONTAL = 1;
VERTICAL = 2;

# merge fragments from two chunks
# @param c0   fragments from first  chunk
# @param c1   fragments from second chunk
# @param sx   (x or y) coordinate of the seam
# @param dr   merge direction, HORIZONTAL or VERTICAL?
# 
def mergeFrags(c0, c1, sx, dr):
  for i in range(len(c1)-1,-1,-1):
    if (dr == HORIZONTAL):
      if (mergeImpl(c0,c1,i,sx,False,1)):continue;
      if (mergeImpl(c0,c1,i,sx,False,3)):continue;
      if (mergeImpl(c0,c1,i,sx,False,0)):continue;
      if (mergeImpl(c0,c1,i,sx,False,2)):continue;
    else:
      if (mergeImpl(c0,c1,i,sx,True,1)):continue;
      if (mergeImpl(c0,c1,i,sx,True,3)):continue;
      if (mergeImpl(c0,c1,i,sx,True,0)):continue;
      if (mergeImpl(c0,c1,i,sx,True,2)):continue;      
    
  c0 += c1


# recursive bottom: turn chunk into polyline fragments;
# look around on 4 edges of the chunk, and identify the "outgoing" pixels;
# add segments connecting these pixels to center of chunk;
# apply heuristics to adjust center of chunk
# 
# @param im   the bitmap image
# @param x    left of   chunk
# @param y    top of    chunk
# @param w    width of  chunk
# @param h    height of chunk
# @return     the polyline fragments
# 
def chunkToFrags(im, x, y, w, h):
  frags = []
  on = False; # to deal with strokes thicker than 1px
  li=-1; lj=-1;
  
  # walk around the edge clockwise
  for k in range(h+h+w+w-4):
    i=0; j=0;
    if (k < w):
      i = y+0; j = x+k;
    elif (k < w+h-1):
      i = y+k-w+1; j = x+w-1;
    elif (k < w+h+w-2):
      i = y+h-1; j = x+w-(k-w-h+3); 
    else:
      i = y+h-(k-w-h-w+4); j = x+0;
    
    if (im[i,j]): # found an outgoing pixel
      if (not on):     # left side of stroke
        on = True;
        frags.append([[j,i],[x+w//2,y+h//2]])
    else:
      if (on):# right side of stroke, average to get center of stroke
        frags[-1][0][0]= (frags[-1][0][0]+lj)//2;
        frags[-1][0][1]= (frags[-1][0][1]+li)//2;
        on = False;
    li = i;
    lj = j;
  
  if (len(frags) == 2): # probably just a line, connect them
    f = [frags[0][0],frags[1][0]];
    frags.pop(0);
    frags.pop(0);
    frags.append(f);
  elif (len(frags) > 2): # it's a crossroad, guess the intersection
    ms = 0;
    mi = -1;
    mj = -1;
    # use convolution to find brightest blob
    for i in range(y+1,y+h-1):
      for j in range(x+1,x+w-1):
        s = \
          (im[i-1,j-1]) + (im[i-1,j]) +(im[i-1,j+1])+\
          (im[i,j-1]  ) +   (im[i,j]) +    (im[i,j+1])+\
          (im[i+1,j-1]) + (im[i+1,j]) +  (im[i+1,j+1]);
        if (s > ms):
          mi = i;
          mj = j;
          ms = s;
        elif (s == ms and abs(j-(x+w//2))+abs(i-(y+h//2)) < abs(mj-(x+w//2))+abs(mi-(y+h//2))):
          mi = i;
          mj = j;
          ms = s;

    if (mi != -1):
      for i in range(len(frags)):
        frags[i][1]=[mj,mi]
  return frags;


# Trace skeleton from thinning result.
# Algorithm:
# 1. if chunk size is small enough, reach recursive bottom and turn it into segments
# 2. attempt to split the chunk into 2 smaller chunks, either horizontall or vertically;
#    find the best "seam" to carve along, and avoid possible degenerate cases
# 3. recurse on each chunk, and merge their segments
# 
# @param im      the bitmap image
# @param x       left of   chunk
# @param y       top of    chunk
# @param w       width of  chunk
# @param h       height of chunk
# @param csize   chunk size
# @param maxIter maximum number of iterations
# @param rects   if not null, will be populated with chunk bounding boxes (e.g. for visualization)
# @return        an array of polylines
# 
def traceSkeleton(im, x, y, w, h, csize, maxIter, rects):
  
  frags = []
  
  if (maxIter == 0): # gameover
    return frags;
  if (w <= csize and h <= csize): # recursive bottom
    frags += chunkToFrags(im,x,y,w,h);
    return frags;
  
  ms = im.shape[0]+im.shape[1]; # number of white pixels on the seam, less the better
  mi = -1; # horizontal seam candidate
  mj = -1; # vertical   seam candidate
  
  if (h > csize): # try splitting top and bottom
    for i in range(y+3,y+h-3):
      if (im[i,x]  or im[(i-1),x]  or im[i,x+w-1]  or im[(i-1),x+w-1]):
        continue
      
      s = 0;
      for j in range(x,x+w):
        s += im[i,j];
        s += im[(i-1),j];
      
      if (s < ms):
        ms = s; mi = i;
      elif (s == ms  and  abs(i-(y+h//2))<abs(mi-(y+h//2))):
        # if there is a draw (very common), we want the seam to be near the middle
        # to balance the divide and conquer tree
        ms = s; mi = i;
  
  if (w > csize): # same as above, try splitting left and right
    for j in range(x+3,x+w-2):
      if (im[y,j] or im[(y+h-1),j] or im[y,j-1] or im[(y+h-1),j-1]):
        continue
      
      s = 0;
      for i in range(y,y+h):
        s += im[i,j];
        s += im[i,j-1];
      if (s < ms):
        ms = s;
        mi = -1; # horizontal seam is defeated
        mj = j;
      elif (s == ms  and  abs(j-(x+w//2))<abs(mj-(x+w//2))):
        ms = s;
        mi = -1;
        mj = j;

  nf = []; # new fragments
  if (h > csize  and  mi != -1): # split top and bottom
    L = [x,y,w,mi-y];    # new chunk bounding boxes
    R = [x,mi,w,y+h-mi];
    
    if (notEmpty(im,L[0],L[1],L[2],L[3])): # if there are no white pixels, don't waste time
      if(rects!=None):rects.append(L);
      nf += traceSkeleton(im,L[0],L[1],L[2],L[3],csize,maxIter-1,rects) # recurse
    
    if (notEmpty(im,R[0],R[1],R[2],R[3])):
      if(rects!=None):rects.append(R);
      mergeFrags(nf,traceSkeleton(im,R[0],R[1],R[2],R[3],csize,maxIter-1,rects),mi,VERTICAL);
    
  elif (w > csize  and  mj != -1): # split left and right
    L = [x,y,mj-x,h];
    R = [mj,y,x+w-mj,h];
    if (notEmpty(im,L[0],L[1],L[2],L[3])):
      if(rects!=None):rects.append(L);
      nf+=traceSkeleton(im,L[0],L[1],L[2],L[3],csize,maxIter-1,rects);
    
    if (notEmpty(im,R[0],R[1],R[2],R[3])):
      if(rects!=None):rects.append(R);
      mergeFrags(nf,traceSkeleton(im,R[0],R[1],R[2],R[3],csize,maxIter-1,rects),mj,HORIZONTAL);
    
  frags+=nf;
  if (mi == -1  and  mj == -1): # splitting failed! do the recursive bottom instead
    frags += chunkToFrags(im,x,y,w,h);
  
  return frags

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='same')
    return y


def getNormal(pt1, pt2, pt3):
  # check if normal is parallel to y-axis
  grad1 = (pt2[0] - pt1[0]) / (pt2[1] - pt1[1])
  grad2 = (pt3[0] - pt2[0]) / (pt3[1] - pt2[1])
  grad = (grad1 + grad2) / 2
  normalGrad = -1 / grad

  return normalGrad

def widthAnalysis(points, im0, n):
  widths = []
  coordsList = []
  for i in range(n, len(points) - n, n):
    coords = points[i]
    coordsNN = points[i + 1]
    coordsPP = points[i - 1]

    gradient = getNormal(coordsPP, coords, coordsNN)

    # check up and down width 
    upwidth = 0
    downwidth = 0

    curcoord = coords

    # check up width
    while im0[curcoord[1]][curcoord[0]].any():
      
      if not math.isnan(gradient) and not math.isinf(gradient):
        print(gradient)
        x2 = curcoord[1] + 1
        y2 = int(gradient * x2 - coords[1] * gradient + coords[0])
      else:
        x2 = curcoord[1]
        y2 = curcoord[0] + 1

      curcoord = [y2, x2]
    
    upcoords = curcoord
    curcoord = coords

    # check down width
    c = (200*random.random(),200*random.random(),200*random.random())
    while im0[curcoord[1]][curcoord[0]].any():

      if not math.isnan(gradient) and not math.isinf(gradient):
        x2 = curcoord[1] - 1
        y2 = int(gradient * x2 - coords[1] * gradient + coords[0])
      else:
        x2 = curcoord[1]
        y2 = curcoord[0] - 1

      # downwidth += math.sqrt((y2 - curcoord[0])**2 + (x2-curcoord[1])**2)
      curcoord = [y2, x2]

    downcoords = curcoord

    totalwidth = math.sqrt((upcoords[0] - downcoords[0])**2 + (upcoords[1] - downcoords[1])**2)
    # cv2.line(im0, (upcoords[0], upcoords[1]), (downcoords[0], downcoords[1]), c, thickness=1)

    widths.append(totalwidth)
    coordsList.append(points[i])

    # arr.append([coords[1], coords[0], gradient, upwidth, downwidth, totalwidth]) 
  
  return widths, coordsList

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

  

if __name__ == "__main__":
  import cv2
  import random

  im0 = cv2.imread("A:/segmented/2477_30_lcx2bin_mask.png")
  imSegmented = cv2.imread("A:/segmented/2477_30_lcx2segmented_threshold_binary.png")

  im = (im0[:,:,0]>128).astype(np.uint8)

  # for i in range(im.shape[0]):
  #   for j in range(im.shape[1]):
  #     print(im[i,j],end="")
  #   print("")
  # print(np.sum(im),im.shape[0]*im.shape[1])
  im = thinning(im);


  rects = []
  
  polys = traceSkeleton(im,0,0,im.shape[1],im.shape[0],10,999,rects)
  
  points = getAllPoints(polys)
  print(points)

  arr, coordsList = widthAnalysis(points, imSegmented, 1)
  arr = np.array(arr)
  arr_s = smooth(arr, 22)
  average_width = np.average(arr)

  peaks, properties = find_peaks(np.negative(arr_s), distance=5, prominence=(average_width*0.4, None))
  
  # plt.plot(range(1, len(arr) + 1), arr)
  plt.plot(range(1, len(arr_s) + 1), arr_s)
  plt.plot(peaks, arr_s[peaks], "x")
  plt.show()

  print(arr, len(arr_s), len(coordsList), len(arr))
  
  circleIm = imSegmented
  for peak in peaks:
    print(int((peak / len(peaks)) * len(coordsList)))
    c = (200*random.random(),200*random.random(),200*random.random())
    cv2.circle(circleIm, coordsList[int((peak / len(arr_s)) * len(coordsList))], 2, c, 2)

  # cv2.imshow('',circleIm);cv2.waitKey(0)
  cv2.imshow('',imSegmented);cv2.waitKey(0)

  for l in polys:
      c = (200*random.random(),200*random.random(),200*random.random())
      for i in range(0,len(l)-1):
        cv2.line(imSegmented,(l[i][0],l[i][1]),(l[i+1][0],l[i+1][1]), c, thickness=2)

  cv2.imshow('',imSegmented);cv2.waitKey(0)

