import pandas as pd
import numpy as np
from pathlib import Path
import math
from skeletonization import getScore

data = pd.read_csv("B://percentage_stenosis.csv")

data = pd.DataFrame(data)

pathString = 'A://segmented'
path = Path(pathString)

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

    for keyframe in video.iterdir():
        for artery in valid_arteries:
            filename = keyframe.name + '_' + artery
            _, scores = getScore(filename, folderDirectory='A:/segmented/', show=False)
            for key, score in scores.items():
                segmentName = artery + '_' + key
                if segmentName in valid_segments:
                    stenosisPercentages[segmentName].append(score)

    print(stenosisPercentages)

