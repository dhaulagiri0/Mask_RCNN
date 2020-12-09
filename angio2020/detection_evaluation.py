import pandas as pd
import numpy as np
from pathlib import Path
import math
from skeletonization import getScore
import os
import json

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def getMetrics(results):
    # calculates sensitivity, positive predictive value and f1 score
    tp = results['tp']
    fp = results['fp']
    fn = results['fn']

    if tp + fn !=0 and tp + fp != 0 and tp + 0.5*(fp + fn) != 0:
        sn = tp / (tp + fn)
        ppv = tp / (tp + fp)
        f1 = tp / (tp + 0.5*(fp + fn))
        return sn, ppv, f1
    return None, None, None

def bboxScore(y_true_list, y_pred, iouThresh):
    for i in range(len(y_true_list)):
        iou = get_iou(y_true_list[i], y_pred)
        if iou > iouThresh:
            return 'tp', i
    return 'fp', None

def processBbox(jsonPath, y_preds, artery, results, iouThresh = 0.5):
    y_true_list = []
    with open(jsonPath) as json_file:
        data = json.load(json_file)
        for region in data['regions']:
            if region['tags'][0] == artery:
                topLeft = region['points'][0]
                bottomRight = region['points'][2]
                y_true_list.append({'x1': topLeft['x'], 'y1': topLeft['y'], 'x2': bottomRight['x'], 'y2': bottomRight['y']})
    
    matched = []
    if len(y_true_list) > 0:
        for y_pred in y_preds:
            result, bbox_matched = bboxScore(y_true_list, y_pred, iouThresh)
            if bbox_matched != None and bbox_matched not in matched:
                matched.append(bbox_matched)
            results[result] += 1
    else:
        results['fp'] += len(y_preds)
    
    results['fn'] += len(y_true_list) - len(matched)

if __name__ == "__main__":
    import argparse

     # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate segmentation masks of arteries')
    parser.add_argument('--segmented_path', required=False,
                        default='A:/segmented',
                        metavar="/path/to/segmented masks",
                        help="Path to folder containing images and their segmentation maps (default: A:/segmented)")


    args = parser.parse_args()
    pathString = args.segmented_path
    path = Path(pathString)

    metricsDict = {
        'sn': None,
        'ppv': None,
        'f1': None
    }

    results = {
        'fp': 0,
        'fn': 0,
        'tp': 0
    }

    for video in path.iterdir():
        for keyframe in video.iterdir():
            for f in keyframe.iterdir():
                if '_bin_mask' in f.name:
                    artery = f.name.split('_')[-3]
                    filename = keyframe.name + '_' + artery
                    jsonPath = f"{pathString.split('/')[0]}/bbox_json/{filename.split('.')[0].split('_')[0]}_{filename.split('.')[0].split('_')[1]}.json"
                    if os.path.exists(jsonPath):
                        _, scores, boxes = getScore(filename, folderDirectory=pathString, show=False, save=True)
                        print(boxes)
                        if boxes != None:
                            processBbox(f"{pathString.split('/')[0]}/bbox_json/{filename.split('.')[0].split('_')[0]}_{filename.split('.')[0].split('_')[1]}.json", boxes, artery, results)
                        print('processed ' + f.name)
    
    sn, ppv, f1 = getMetrics(results)
    print(f'Mean Sensitivity: {sn}, Mean Positive Predictive Rate: {ppv}, Mean F1 Score: {f1}')

