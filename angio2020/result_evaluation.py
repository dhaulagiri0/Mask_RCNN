import pandas as pd
import numpy as np
from pathlib import Path

data = pd.read_csv("A://percentage_stenosis.csv")

data = pd.DataFrame(data)


path = Path('B://segmented')

for video in path.iterdir():
    row = data.loc[data['keyframe_id'] == float(video.name)].head()
    keyframes = []
    valid_arteries = []
    for keyframe in video.iterdir():
        keyframes.append(keyframe.name)  
    for index, value in row.items():
        print(type(value.isna()))
        for i, v in value.isna().items():
            print(i, v)
                

