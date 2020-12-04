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



data = pd.read_csv("B:/percentage_stenosis.csv")

data = pd.DataFrame(data)

pathString = 'A:/segmented'
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
            # folderDirectory = pathString
            if os.path.exists(f"{pathString}/{filename.split('_')[0]}/{filename.split('.')[0].split('_')[0]}_{filename.split('.')[0].split('_')[1]}/{filename}bin_mask.png"):
                _, scores = getScore(filename, folderDirectory=pathString, show=False)
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


"""
mean errors first round:
raw percentages {'lad_p': None, 'lad_m': None, 'lad_d': None, 'lcx2_p': None, 'lcx2_m': None, 'lcx2_d': None, 'diagonal': None, 'lcx1': None}
raw percentages {'lad_p': 41.313015853693784, 'lad_m': None, 'lad_d': None, 'lcx2_p': None, 'lcx2_m': None, 'lcx2_d': None, 'diagonal': None, 'lcx1': None}
raw percentages {'lad_p': 10.966248934731032, 'lad_m': None, 'lad_d': None, 'lcx2_p': None, 'lcx2_m': None, 'lcx2_d': None, 'diagonal': None, 'lcx1': None}
raw percentages {'lad_p': None, 'lad_m': 26.066242398934016, 'lad_d': None, 'lcx2_p': None, 'lcx2_m': None, 'lcx2_d': None, 'diagonal': None, 'lcx1': None}
raw percentages {'lad_p': 8.003781948507216, 'lad_m': 19.121593196430613, 'lad_d': None, 'lcx2_p': None, 'lcx2_m': None, 'lcx2_d': None, 'diagonal': None, 'lcx1': None} 
raw percentages {'lad_p': 38.77428614929597, 'lad_m': 43.85646968075416, 'lad_d': 12.041105703580937, 'lcx2_p': 75.9823148125468, 'lcx2_m': None, 'lcx2_d': None, 'diagonal': None, 'lcx1': None}
raw percentages {'lad_p': 11.854085439915279, 'lad_m': 89.22776182014374, 'lad_d': None, 'lcx2_p': None, 'lcx2_m': None, 'lcx2_d': None, 'diagonal': None, 'lcx1': None} 

raw errors (%):  {'lad_p': [46.686892761629004, 20.965261803413, 64.41693059993003, 79.77501597606043, 76.21308494070821, 48.358730182882766, 84.33393009324138, 89.32829073532372, 44.608162643862904, 86.82879395564969], 'lad_m': [6.831127032247775, 52.65183269808609, 44.22579874192373, 85.52712933460606, 83.0061080784261, 76.47130541090596, 30.331211994670078, 52.19601700892347, 45.179412899057304, 18.970349093524987], 'lad_d': [67.06007610999922, 36.769424627733585, 42.50495876832693, 75.91778859283812], 
'lcx2_p': [46.05357599877789, 8.546164017924006], 'lcx2_m': [], 'lcx2_d': [], 'diagonal': [], 'lcx1': []}
mean errors (%):  {'lad_p': 64.15150936927012, 'lad_m': 49.53902922923716, 'lad_d': 55.56306202472446, 'lcx2_p': 27.299870008350947, 'lcx2_m': None, 'lcx2_d': None, 'diagonal': None, 'lcx1': None}
"""


