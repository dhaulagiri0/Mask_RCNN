from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from pathlib import Path
import cv2
import numpy as np
import imageio
import os, json
from shutil import copyfile

def getPolyImage(points_dict, shape=(512, 512)):
    points = []
    for point in points_dict:
        points.append([point['x'], point['y']])
    points = np.array(points).reshape((-1,1,2))
    blankImage = np.zeros(shape=shape)
    cv2.fillPoly(blankImage, np.int32([points]), (255))
    return blankImage

pathString = 'A:/poly_json/'
path = Path(pathString)

for jf in path.iterdir():
    image_id = jf.name.split('.')[0]
    keyframe_folder = 'A:/segmented_manual/' + image_id.split('_')[0] + '/' + image_id + '/'
    with open(jf) as json_file:
        data = json.load(json_file)
        for region in data['regions']:
            artery = region['tags'][0]
            save_name = image_id + '_' + artery + '_segmented_threshold_binary.png'
            points_dict = region['points']
            img = getPolyImage(points_dict)
            if not os.path.exists('A:/segmented_manual/' + image_id.split('_')[0] + '/'):
                os.mkdir('A:/segmented_manual/' + image_id.split('_')[0] + '/')
            if not os.path.exists(keyframe_folder):
                os.mkdir(keyframe_folder)
            imageio.imwrite(keyframe_folder + save_name, img)
            imageio.imwrite(keyframe_folder + image_id + '_' + artery + '_bin_mask.png', img)

    if os.path.exists('A:/test/png/' + image_id + '.png'):
        copyfile('A:/test/png/' + image_id + '.png', keyframe_folder + image_id + '_original.png')
    elif os.path.exists('A:/val/png/' + image_id + '.png'):
        copyfile('A:/val/png/' + image_id + '.png', keyframe_folder + image_id + '_original.png')
