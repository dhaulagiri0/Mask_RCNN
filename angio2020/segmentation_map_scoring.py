from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from pathlib import Path
import cv2
import numpy as np
import os, json

def f1Score(y_true, y_pred):
    # binarise input image
    y_true = np.where(y_true > 0, 1, 0)
    y_pred = np.where(y_pred > 0, 1, 0)
    # convert to 1d as per sklearn requirements
    y_true = y_true.ravel() 
    y_pred = y_pred.ravel()

    # get confusion matrix and values
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    f1 = f1_score(y_true, y_pred)
    return (tn, fp, fn, tp), f1

# gets the json file for one keyframe and returns the required polygon points
def processMaskJson(jsonPath, artery):
    points_dict = {}
    with open(jsonPath) as json_file:
        data = json.load(json_file)
        for region in data['regions']:
            if region['tags'][0] == artery:
                points_dict = region['points']
    return points_dict

def getPolyImage(points_dict, shape=(512, 512)):
    points = []
    for point in points_dict:
        points.append([point['x'], point['y']])
    points = np.array(points).reshape((-1,1,2))
    blankImage = np.zeros(shape=shape)
    cv2.fillPoly(blankImage, np.int32([points]), (255))
    return blankImage

def scoring(pathString):
    path = Path(pathString)
    scores = []
    accuracies = []
    sensitivities = []
    ppvs = []
    # check through segmentation map directory
    for video in path.iterdir():
        print(video.name)
        for keyframe in video.iterdir():
            print('- ', keyframe.name)
            for image in keyframe.iterdir():
                if 'threshold_binary' in image.name:
                    # image is a binary mask
                    pathPrefix = f'{pathString}/{video.name}/{keyframe.name}/'
                    y_pred = cv2.imread(pathPrefix + f'{image.name}', 0)
                    artery = image.name.split('_')[2]
                    # TODO get segmentation map from coco format json file
                    jsonPath = f"{pathString.split('/')[0]}/poly_json/{keyframe.name}.json"

                    y_true = None
                    if Path.exists(Path(jsonPath)):
                        points_dict = processMaskJson(jsonPath, artery)
                        if len(points_dict) > 0:
                            print('-- ', image.name)
                            y_true = np.int32(getPolyImage(points_dict))
                    # else:
                    #     print('-- ', image.name)
                    #     imagePath = f'A:/segmented_otsu/{keyframe.name.split("_")[0]}/{keyframe.name}/{keyframe.name}_{artery}_bin_mask.png'
                    #     y_true = cv2.imread(imagePath, 0)

                    if  not (y_true is None):
                        values, f1 = f1Score(y_true, y_pred)
                        tn, fp, fn, tp = values
                        accuracy = (tp + tn) / (tp + tn + fp + fn)
                        sensitivity = tp / (tp + fn)
                        ppv = tp / (tp + fp)
                        accuracies.append(accuracy)
                        scores.append(f1)
                        sensitivities.append(sensitivity)
                        ppvs.append(ppv)

    f1Mean = np.average(np.array(scores))
    accuracyMean = np.average(np.array(accuracies))
    snMean = np.average(np.array(sensitivities))
    ppvMean = np.average(np.array(ppvs))
    return f1Mean, accuracyMean, snMean, ppvMean
                        

if __name__ == "__main__":
    import argparse

     # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate segmentation masks of arteries')
    parser.add_argument('--data_path', required=False,
                        default='A:/segmented',
                        metavar="/path/to/segmentation/",
                        help="Directory folder for csv file containing scores (default: A:/segmented)")
    
    args = parser.parse_args()
    data_path = args.data_path

    f1Mean, accuracyMean, snMean, ppvMean = scoring(data_path)
    print('mean f1 score: ', f1Mean)
    print('mean accuracy: ', accuracyMean)
    print('mean sn: ', snMean)
    print('mean ppv: ', ppvMean)


# gt = cv2.imread('A:/segmented/1367/1367_35/1367_35_diagonalbin_mask.png', 0)
# gt = np.where(gt > 127.5, 1, 0)
# gt = gt.ravel()

# pred = cv2.imread('A:/segmented/1367/1367_35/1367_35_diagonalsegmented_threshold_binary.png', 0)
# pred = np.where(pred > 127.5, 1, 0)
# pred = pred.ravel()

# print(np.amax(pred))

# cm = confusion_matrix(gt, pred)
# tn, fp, fn, tp = cm.ravel()
# score = f1_score(gt, pred)

# print(f'true negative: {tn}, false positive: {fp}, false negative: {fn}, true positive: {tp}')
# print(f'f1score: {score}')