import cv2
import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from pycocotools import mask as maskUtils
from skimage.filters import frangi, hessian, gaussian
import skimage
from pathlib import Path
import json
import imageio
from PIL import Image
import os
from skimage import data
from skimage import color
from skimage.filters import meijering, sato, frangi, hessian, gabor_kernel
import matplotlib.pyplot as plt
from skimage.segmentation import flood, flood_fill
from skimage import img_as_ubyte
import sys
from scipy import ndimage as ndi
from result_preprocess import upContrast
from skimage.segmentation import active_contour
from result_preprocess import otsu

# def identity(image, **kwargs):
#     """Return the original image, ignoring any kwargs."""
#     return image


def alternativeSegmentationMethod():
    originalImage = cv2.imread('A:/segmented/1367/1367_35/1367_35_diagonal_original.png')

# def contour(img):
#     s = np.linspace(0, 2*np.pi, 400)
#     r = 100 + 100*np.sin(s)
#     c = 220 + 100*np.cos(s)
#     init = np.array([r, c]).T
#     snake = active_contour(gaussian(img, 3),
#                        init, alpha=0.015, beta=10, gamma=0.001)

#     fig, ax = plt.subplots(figsize=(7, 7))
#     ax.imshow(img, cmap=plt.cm.gray)
#     ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
#     ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
#     ax.set_xticks([]), ax.set_yticks([])
#     ax.axis([0, img.shape[1], img.shape[0], 0])

#     plt.show()

def generateFrangiData(dataPath):
    path = Path(dataPath)
    for subset in path.iterdir():
        print(subset.name)
        if not subset.name in ['train', 'val']: continue
        for item in subset.iterdir():
            print(item.name)
            if not item.is_dir():
                # item is a image
                imageName = item.name
                image = cv2.imread(dataPath + '/' + subset.name +'/' + item.name, 0)
                labImage = upContrast(image)
                processedImage = cv2.cvtColor(labImage, cv2.COLOR_LAB2BGR)
                filtered = frangi(cv2.cvtColor(processedImage, cv2.COLOR_BGR2GRAY), beta=1.0, gamma=0.1)
                filtered = img_as_ubyte(filtered)
                skimage.io.imsave(dataPath + '/' + subset.name +'_new/' + item.name, filtered)

def generateFrangiData_(dataPath):
    path = Path(dataPath)
    for subset in path.iterdir():
        # print(subset.name)
        if not subset.name in ['train', 'val']: continue
        for item in subset.iterdir():
            if not item.is_dir():
                print(item.name)
                # item is a image
                imageName = item.name
                image = cv2.imread(dataPath + '/' + subset.name +'/' + item.name, 0)
                labImage = upContrast(image)
                processedImage = cv2.cvtColor(labImage, cv2.COLOR_LAB2BGR)
                filtered = frangi(cv2.cvtColor(processedImage, cv2.COLOR_BGR2GRAY), beta=1.0, gamma=0.1)
                filtered = img_as_ubyte(filtered)
                skimage.io.imsave(dataPath + '/' + subset.name +'_new/' + item.name, filtered)
            break
        break



# image = cv2.imread('A:/test/1578_035.jpeg')
# labImage = upContrast(image)
# processedImage = cv2.cvtColor(labImage, cv2.COLOR_LAB2BGR)
# filtered = frangi(cv2.cvtColor(processedImage, cv2.COLOR_BGR2GRAY), beta=1.0, gamma=0.1)
# imageio.imsave('E:/Documents/angio_plots/frangi_2.png', filtered)

from skeletonization import skeletoniseSkimg

mask = cv2.imread('A:/segmented_otsu/1367/1367_35/1367_35_lcx1_bin_mask.png')
segmask = cv2.imread('A:/segmented_otsu/1367/1367_35/1367_35_lcx1_segmented_threshold_binary.png')
points = skeletoniseSkimg('A:/segmented_otsu/1367/1367_35/1367_35_lcx1_bin_mask.png')

for pt in points:
    segmask[pt[0], pt[1] + 5] = (255, 0, 0)

# cv2.imshow('', mask); cv2.waitKey(0)
cv2.imshow('', segmask); cv2.waitKey(0)
cv2.imwrite('E:/Documents/angio_plots/skeletonised.png', segmask)
    

# nonzero = np.array(filtered[filtered > 0])
# otsu_thresh = otsu(nonzero, is_reduce_noise=True)

# ret, filtered = cv2.threshold(filtered , otsu_thresh, 255, cv2.THRESH_TOZERO_INV)

# filtered = img_as_ubyte(filtered)

# image_f = hessian(image, sigmas=[1, 1.5], sca black_ridges=False)

# plt.imshow(image_f)
# plt.show()

# final_image = np.multiply(image_f, image).astype(np.uint8)
# imageio.imwrite('A:/test.png', final_image)
# segmented_binary = cv2.threshold(np.float32(cv_image, 5, 255, cv2.THRESH_BINARY)
# # print(segmented_binary)

# cv2.imwrite('A:/asd.png', segmented_binary)
# segmented_binary = cv2.resize(segmented_binary, (segmented_binary.shape[0]*2, segmented_binary.shape[1]*2), interpolation=cv2.INTER_AREA)
# kernel = np.ones((2,2),np.uint8)
# segmented_binary = cv2.morphologyEx(segmented_binary, cv2.MORPH_OPEN, kernel, iterations=4)
# segmented_binary = cv2.morphologyEx(segmented_binary, cv2.MORPH_CLOSE, kernel, iterations=4)
# segmented_binary = cv2.resize(segmented_binary, (int(segmented_binary.shape[0]*0.5), int(segmented_binary.shape[1]*0.5)), interpolation=cv2.INTER_AREA)



# fig, axes = plt.subplots(2, 5)
# for i, black_ridges in enumerate([1, 0]):
#     for j, func in enumerate([identity, meijering, sato, frangi, hessian]):
#         kwargs['black_ridges'] = black_ridges
#         result = func(image, **kwargs)
#         axes[i, j].imshow(result, cmap=cmap, aspect='auto')
#         if i == 0:
#             axes[i, j].set_title(['Original\nimage', 'Meijering\nneuriteness',
#                                   'Sato\ntubeness', 'Frangi\nvesselness',
#                                   'Hessian\nvesselness'][j])
#         if j == 0:
#             axes[i, j].set_ylabel('black_ridges = ' + str(bool(black_ridges)))
#         axes[i, j].set_xticks([])
#         axes[i, j].set_yticks([])

# plt.tight_layout()
# plt.show()