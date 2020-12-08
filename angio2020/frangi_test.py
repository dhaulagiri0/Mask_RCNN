import cv2
import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from pycocotools import mask as maskUtils
from skimage.filters import frangi, hessian
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

# def identity(image, **kwargs):
#     """Return the original image, ignoring any kwargs."""
#     return image

dataPath = 'D:/'
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
            meanPixel = np.average(image)
            print(meanPixel)
            # labImage = upContrast(image)
            # processedImage = cv2.cvtColor(labImage, cv2.COLOR_LAB2BGR)
            # filtered = frangi(cv2.cvtColor(processedImage, cv2.COLOR_BGR2GRAY), beta=1.0, gamma=0.1)
            # filtered = img_as_ubyte(filtered)
            # skimage.io.imsave(dataPath + '/' + subset.name +'_new/' + item.name, filtered)
# plt.imshow(results[0])
# plt.show

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