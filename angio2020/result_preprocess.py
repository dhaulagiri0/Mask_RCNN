import cv2
import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from pycocotools import mask as maskUtils
from pathlib import Path
import json
import imageio
from PIL import Image

def toBinMask(img=None, path=None, threshold=10):
    if path:
        # read mask in grayscale format
        img = cv2.imread(path, 0)

    # create binary map by thresholding
    ret, binMap = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    #convert bin map to rle
    return binMap

def upContrast(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))

    #-----Converting image from LAB Color model to RGB model--------------------
    lab = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    final = cv2.cvtColor(lab, cv2.COLOR_RGB2GRAY)
    return final

    #_____END_____#

def makeSegmentations(data_path, subset, save_path):
    p = Path(data_path)
    # navigates the items found in data folder
    for item in p.iterdir():
        # if item is a folder containing masks
        if item.is_dir():
            # image_id is the folder name
            image_id = item.name
            # iterate through all the masks
            for f in item.iterdir():
                # get binary mask
                # binMask = np.expand_dims(toBinMask(path=str(f)), -1)
                binMask = toBinMask(path=str(f))

                originalImage = cv2.imread(data_path + '/png/' + image_id + '.png')
                originalImage = upContrast(originalImage)

                # isolate masked region
                segmented = cv2.bitwise_and(originalImage, originalImage, mask=binMask)
                segmented_v = segmented

                # find sum of pixel value
                sumPx = np.sum(segmented)

                # count occurences of pixel value above 0 this also doubles as masked area
                occ = (segmented > 0).sum()

                # mean pixel value
                meanPx = sumPx / occ

                # threshold segmented image by mean pixel value
                ret, segmented_thresh = cv2.threshold(segmented , meanPx * 1.2, 0, cv2.THRESH_TOZERO_INV)

                im_floodfill = segmented_thresh.copy()
                h, w = segmented_thresh.shape[:2]
                mask = np.zeros((h+2, w+2), np.uint8)
                cv2.floodFill(im_floodfill, mask, (0,0), 255)
                im_floodfill_inv = cv2.bitwise_not(im_floodfill)
                segmented_binary = segmented_thresh | im_floodfill_inv

                # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
                # segmented_thresh = cv2.morphologyEx(segmented_thresh, cv2.MORPH_OPEN, kernel, iterations=1)

                # binary map
                # ret, segmented_binary = cv2.threshold(segmented_thresh, 1, 255, cv2.THRESH_BINARY )

                # find area percentage
                arteryArea = (segmented_binary > 0).sum()
                percentage = arteryArea / occ

                # save
                imageio.imwrite(save_path + f.name.split('.')[0] + 'bin_mask.png', binMask)
                imageio.imwrite(save_path + f.name.split('.')[0] + 'segmented.png', segmented_v)
                imageio.imwrite(save_path + f.name.split('.')[0] + 'segmented_threshold.png', segmented_thresh)
                imageio.imwrite(save_path + f.name.split('.')[0] + 'segmented_threshold_binary.png', segmented_binary)
                print(f.name.split('.')[0] + ' area percentage: ' + str(percentage))

makeSegmentations('A:/val', 'val', 'A:/segmented/')