import cv2
import numpy as np
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from pycocotools import mask as maskUtils
from skimage.filters import hessian
from pathlib import Path
import json
import imageio
from PIL import Image
import os
from skimage.segmentation import flood, flood_fill

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

    # Splitting the LAB image to different channels
    l, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(5,5))
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to RGB model
    lab = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return lab

def hessianFiltering(image, fileprefix):
    # img must be read with skimg
    image_f = hessian(image, sigmas=[1], black_ridges=False)
    final_image = np.multiply(image_f, image).astype(np.uint8)
    imageio.imwrite(f'{fileprefix}_hessian_filtered.png', final_image)
    image = cv2.imread(f'{fileprefix}_hessian_filtered.png', 0)
    return cv2.imread(f'{fileprefix}_hessian_filtered.png', 0)


def otsu(image, is_normalized=True, is_reduce_noise=False):

    # Apply GaussianBlur to reduce image noise if it is required
    if is_reduce_noise:
        image = cv2.GaussianBlur(image, (5, 5), 0)

    # Set total number of bins in the histogram
    bins_num = 256

    # Get the image histogram
    hist, bin_edges = np.histogram(image, bins=bins_num)

    # Get normalized histogram if it is required
    if is_normalized:
        hist = np.divide(hist.ravel(), hist.max())

    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.

    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1
    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)

    threshold = bin_mids[:-1][index_of_max_val]
    return threshold

def makeSegmentations(data_path, subset, save_path, hess=False):
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
                if not os.path.exists(data_path + '/png/' + image_id + '.png'):
                    if not os.path.exists(data_path + '/png'): os.mkdir(data_path + '/png')
                    originalImage = cv2.imread(data_path + '/' + image_id + '.jpeg')
                    cv2.imwrite(data_path + '/png/' + image_id + '.png', originalImage)

                # path for saving processed image files
                filePrefix = f"{save_path}/{image_id.split('_')[0]}/{image_id}/{f.name.split('.')[0]}"
                
                # create save paths
                if not os.path.exists(f"{save_path}/{image_id.split('_')[0]}"):
                    os.mkdir(f"{save_path}/{image_id.split('_')[0]}")
                if not os.path.exists(f"{save_path}/{image_id.split('_')[0]}/{image_id}"):
                    os.mkdir(f"{save_path}/{image_id.split('_')[0]}/{image_id}")

                originalImage = cv2.imread(data_path + '/png/' + image_id + '.png')
                upCon = upContrast(originalImage.copy())
                upCon = cv2.cvtColor(upCon, cv2.COLOR_RGB2GRAY)

                # isolate masked region
                segmented = cv2.bitwise_and(upCon, upCon, mask=binMask)
                segmented_v = segmented

                # find sum of pixel value
                sumPx = np.sum(segmented)

                # count occurences of pixel value above 0 this also doubles as masked area
                occ = (segmented > 0).sum()

                # mean pixel value
                meanPx = sumPx / occ


                if hess:
                    segmented_thresh = hessianFiltering(segmented, filePrefix)
                else:
                    # Otsu Thresholding
                    # get only non-zero pixel values to select otsu threshold
                    nonzero = np.array(segmented[segmented > 0])
                    otsu_thresh = otsu(upCon, is_reduce_noise=True)

                    # threshold segmented image by otsu value
                    ret, segmented_thresh = cv2.threshold(segmented , otsu_thresh, 0, cv2.THRESH_TOZERO_INV)

                # threshold segmented image by mean
                # ret, segmented_thresh = cv2.threshold(segmented , meanPx*1.2, 0, cv2.THRESH_TOZERO_INV)

                im_floodfill = segmented_thresh.copy()
                h, w = segmented_thresh.shape[:2]
                mask = np.zeros((h+2, w+2), np.uint8)
                cv2.floodFill(im_floodfill, mask, (0,0), 255)
                im_floodfill_inv = cv2.bitwise_not(im_floodfill)
                segmented_binary = segmented_thresh | im_floodfill_inv

                # # binary map
                # ret, segmented_binary = cv2.threshold(segmented_thresh, 1, 255, cv2.THRESH_BINARY )

                # morphology opening and closing
                segmented_binary = cv2.resize(segmented_binary, (segmented_binary.shape[0]*2, segmented_binary.shape[1]*2), interpolation=cv2.INTER_AREA)
                if hess:
                    kernel = np.ones((4,4),np.uint8)
                    segmented_binary = cv2.morphologyEx(segmented_binary, cv2.MORPH_CLOSE, kernel, iterations=4)
                else:
                    kernel = np.ones((2,2),np.uint8)
                    segmented_binary = cv2.morphologyEx(segmented_binary, cv2.MORPH_OPEN, kernel, iterations=4)
                    segmented_binary = cv2.morphologyEx(segmented_binary, cv2.MORPH_CLOSE, kernel, iterations=4)
                segmented_binary = cv2.resize(segmented_binary, (int(segmented_binary.shape[0]*0.5), int(segmented_binary.shape[1]*0.5)), interpolation=cv2.INTER_AREA)

                # save
                imageio.imwrite(filePrefix + 'bin_mask.png', binMask)
                imageio.imwrite(filePrefix + 'segmented.png', segmented_v)
                # imageio.imwrite(save_path + f.name.split('.')[0] + 'segmented_blur.png',blur)
                imageio.imwrite(filePrefix + 'segmented_threshold_binary.png', segmented_binary)
                imageio.imwrite(filePrefix + 'segmented_threshold.png', segmented_thresh)
                # print(f.name.split('.')[0] + ' area percentage: ' + str(percentage))
                print(f'processed: {f.name}')


if __name__ == "__main__":
    import argparse

     # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate segmentation masks of arteries')
    parser.add_argument('--destination_path', required=False,
                        default='A:/segmented/',
                        metavar="/path/to/angio/",
                        help="Directory folder for generated masks (default: A:/segmented)")
    parser.add_argument('--subset_folder', required=False,
                        default='A:/test/',
                        metavar="/path/to/subset",
                        help="Path to whichever subset will be used to generate the segmentation maps: train, test or val. Default: A:/test")
    parser.add_argument('--mode', required=False,
                    default='otsu',
                    metavar="bool",
                    help="thresholding mode to use: hess or otsu (default)")


    args = parser.parse_args()
    destPath = args.destination_path
    subsetPath = args.subset_folder
    subset = subsetPath.split('/')[-1]
    mode = args.mode
    if mode == 'hess':
        makeSegmentations(subsetPath, subset, destPath, hess=True)
    else:
        makeSegmentations(subsetPath, subset, destPath, hess=False)
