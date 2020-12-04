import pandas as pd
import numpy as np
from pathlib import Path
import math
from skeletonization import getScore
import os

# compute average given a dictionary of arrays containing percentages or percentage error
def averagePerSegment(percDict):
    for segment, percentages in percDict.items():
        sumPercentage = 0
        average = None
        if len(percentages) > 0:
            for percentage in percentages:
                sumPercentage += percentage
            average = sumPercentage / len(percentages) 
        percDict[segment] = average

def calculateErrors(predDict, gtDict, errorsDict):
    for segment, percentage in predDict.items():
        gt = gtDict[segment]
        if percentage != None and gt != None:
            error = abs((percentage - gt)/ gt) * 100
            errorsDict[segment].append(error)



data = pd.read_csv("A://percentage_stenosis.csv")

data = pd.DataFrame(data)

pathString = 'B://segmented'
path = Path(pathString)

errors = {
    'lad_p': [],
    'lad_m': [],
    'lad_d': [],
    'lcx2_p': [],
    'lcx2_m': [],
    'lcx2_d': [],
    'diagonal' : [],
    'lcx1' : []
}

for video in path.iterdir():
    row = data.loc[data['keyframe_id'] == float(video.name)].head()
    stenosisPercentages = {
        'lad_p': [],
        'lad_m': [],
        'lad_d': [],
        'lcx2_p': [],
        'lcx2_m': [],
        'lcx2_d': [],
        'diagonal' : [],
        'lcx1' : []
    }
    gtPercentages = {
        'lad_p': None,
        'lad_m': None,
        'lad_d': None,
        'lcx2_p': None,
        'lcx2_m': None,
        'lcx2_d': None,
        'diagonal' : None,
        'lcx1' : None
    }
    if len(row) > 0:
        row = row.iloc[0]
    else:
        continue
        # print(row)
    valid_arteries = []
    valid_segments = []
    for index, value in row.items():
        # print(value)
        if not math.isnan(value) and index != 'keyframe_id':
            artery = index.split('_')[0]
            if artery not in valid_arteries:
                valid_arteries.append(artery)
            valid_segments.append(index)
            gtPercentages[index] = value

    for keyframe in video.iterdir():
        for artery in valid_arteries:
            filename = keyframe.name + '_' + artery
            folderDirectory = 'B:/segmented/'
            if os.path.exists(f"{folderDirectory}/{filename.split('_')[0]}/{filename.split('.')[0].split('_')[0]}_{filename.split('.')[0].split('_')[1]}/{filename}bin_mask.png"):
                _, scores = getScore(filename, folderDirectory='B:/segmented/', show=False)
                for key, score in scores.items():
                    segmentName = artery + '_' + key
                    if segmentName in valid_segments:
                        stenosisPercentages[segmentName].append(score)
    
    averagePerSegment(stenosisPercentages)
    print('raw percentages', stenosisPercentages)
    calculateErrors(stenosisPercentages, gtPercentages, errors)

print('raw errors: ', errors)
averagePerSegment(errors)
print('mean errors: ', errors)
    



