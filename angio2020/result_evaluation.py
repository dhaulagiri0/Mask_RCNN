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
                print(f'{filename}: ' ,scores)
                for key, score in scores.items():
                    segmentName = artery + '_' + key
                    if artery == 'lcx1' or artery == 'diagonal':
                        segmentName = artery
                    if segmentName in valid_segments:
                        stenosisPercentages[segmentName].append(score)
    
    averagePerSegment(stenosisPercentages)
    print(f'{video.name}: raw percentages', stenosisPercentages)
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

"""
with precise ptsalongline
1388: raw percentages {'lad_p': None, 'lad_m': 45.55553041861184, 'lad_d': 20.537392472901853, 'lcx2_p': None, 'lcx2_m': None, 'lcx2_d': None, 'diagonal': None, 'lcx1': 
None}
1472: raw percentages {'lad_p': 67.84798434973884, 'lad_m': 29.153827940409982, 'lad_d': None, 'lcx2_p': 78.96524718768373, 'lcx2_m': None, 'lcx2_d': None, 'diagonal': 50.04116259608066, 'lcx1': None}
1494: raw percentages {'lad_p': 100.0, 'lad_m': 49.87339736767986, 'lad_d': None, 'lcx2_p': None, 'lcx2_m': None, 'lcx2_d': None, 'diagonal': None, 'lcx1': 95.30264490589377}
1523: raw percentages {'lad_p': 58.83194438693817, 'lad_m': 10.9725814722862, 'lad_d': 66.3299211927877, 'lcx2_p': None, 'lcx2_m': None, 'lcx2_d': None, 'diagonal': None, 'lcx1': None}



details:
1388:
    1388_42_lad:  {'p': 10.31461492214607, 'm': 39.05347955041552, 'd': 7.070041516003778}
    1388_43_lad:  {'p': 11.114385412162731, 'm': 22.848833435839786, 'd': 47.259774124454566}
    1388_44_lad:  {'p': 10.864882225303385, 'm': 22.66940852471223, 'd': 27.18461463699098}
    1388_45_lad:  {'p': 12.144283688016932, 'm': 41.99734767939408, 'd': 32.62050499784325}
    1388_46_lad:  {'p': 8.94778266147136, 'm': 28.31267366717646, 'd': 7.483372807659916}
    1388_47_lad:  {'p': 12.990039184386305, 'm': 73.88986270540438, 'd': 19.47120705409099}
    1388_48_lad:  {'p': 18.367331260851095, 'm': 27.311326986088027, 'd': 12.765693606144612}
    1388_49_lad:  {'p': 10.616028849471348, 'm': 33.04894729482325, 'd': 13.775387527886906}
    1388_50_lad:  {'p': 8.352104809081728, 'm': 28.26756584999649, 'd': 14.159105360986246}
    1388_51_lad:  {'p': 8.012074097653842, 'm': 79.00132282957772, 'd': 12.232427737202634}
    1388_52_lad:  {'p': 9.327743902438968, 'm': 88.73814425276211, 'd': 7.930183133912882}
    1388_53_lad:  {'p': 7.12027714773793, 'm': 98.53260340048546, 'd': 32.575955192957714}
    1388_54_lad:  {'m': 18.37663910120062, 'd': 11.259582501788357}
    1388_55_lad:  {'p': 9.28887529011413, 'm': 18.672992457042824, 'd': 36.94532894381803}
    1388_56_lad:  {'p': 5.6182522958453385, 'm': 27.85026908352287, 'd': 15.269912142441711}
    1388_57_lad:  {'p': 17.604300526775283, 'm': 16.831273651632582, 'd': 24.503441424736504}
    1388_58_lad:  {'p': 5.157712288360672, 'm': 30.892917556345644, 'd': 37.96761380246478}
    1388_59_lad:  {'p': 4.267821031912311, 'm': 12.144261023611602, 'd': 6.211406403223785}
    1388_60_lad:  {'p': 3.667638748569957, 'm': 72.56781124617349, 'd': 32.31447411170011}
    1388_61_lad:  {'p': 5.5535856100856655, 'm': 95.39400800941881, 'd': 17.042407134150118}
    1388_62_lad:  {'p': 22.59772279068687, 'm': 80.26445048522449, 'd': 15.24280777048107}
1472:
    1472_049_lad:  {'p': 44.39094877570493, 'm': 22.116977940414884, 'd': 8.839699074074158}
    1472_049_diagonal:  {'diagonal': 78.96524718768373}
    1472_049_lcx2:  {'p': 78.96524718768373, 'd': 21.226132500229745}
    1472_050_lad:  {'p': 22.14411777107155, 'm': 21.46622291051451}
    1472_050_diagonal:  {'diagonal': 86.79020801909026}
    1472_051_lad:  {'p': 37.85860090815749, 'm': 38.810976111010845, 'd': 10.161699542171675}
    1472_051_diagonal:  {'diagonal': 28.228942276096337}
    1472_059_lad:  {'p': 19.18781916374015, 'm': 100.0}
    1472_059_diagonal:  {'diagonal': 36.593765843254666}
    1472_060_lad:  {'p': 100.0, 'm': 7.12897050461031, 'd': 44.22664557605882}
    1472_060_diagonal:  {'diagonal': 11.124240306630206}
    1472_061_lad:  {'p': 100.0, 'm': 45.14772597449043, 'd': 4.65368143860061}
    1472_061_diagonal:  {'diagonal': 16.32523767550219}
    1472_062_lad:  {'p': 87.05037252897527, 'm': 8.225742047118278, 'd': 99.54796274919234}
    1472_062_diagonal:  {'diagonal': 60.41062292478442}
    1472_063_lad:  {'p': 100.0, 'm': 7.019697119667356, 'd': 100.0}
    1472_063_diagonal:  {'diagonal': 31.932199131684126}
    1472_065_lad:  {'p': 100.0, 'm': 12.46813885586321, 'd': 100.0}
    1472_065_diagonal:  {'diagonal': 100.0}
1494:
    1494_054_lad:  {'p': 100.0, 'm': 23.31524350330062, 'd': 55.04247693128943}
    1494_054_lcx1:  {'lcx1': 85.46260346907081}
    1494_055_lad:  {'p': 100.0, 'm': 6.798318585873686, 'd': 12.46036510140267}
    1494_055_lcx1:  {'lcx1': 94.49056158796502}
    1494_056_lad:  {'p': 100.0, 'd': 9.751025080206189}
    1494_056_lcx1:  {'lcx1': 100.0}
    1494_057_lad:  {'p': 100.0, 'm': 81.06070118984464, 'd': 42.32277809006318}
    1494_057_lcx1:  {'lcx1': 100.0}
    1494_058_lad:  {'p': 100.0, 'm': 100.0}
    1494_058_lcx1:  {'lcx1': 100.0}
    1494_059_lad:  {'p': 100.0, 'm': 76.51605304653393, 'd': 11.638927398037813}
    1494_059_lcx1:  {'lcx1': 88.8482339446594}
    1494_060_lad:  {'p': 100.0, 'm': 100.0, 'd': 5.7693182981918945}
    1494_060_lcx1:  {'lcx1': 74.8303398690299}
    1494_061_lad:  {'p': 100.0, 'm': 58.4445312722796, 'd': 5.093210506697132}
    1494_061_lcx1:  {'lcx1': 100.0}
    1494_062_lad:  {'p': 100.0, 'm': 69.78450955223062, 'd': 18.134067603494085}
    1494_062_lcx1:  {'lcx1': 100.0}
    1494_063_lad:  {'p': 100.0, 'm': 10.41807102757939, 'd': 11.82229199034478}
    1494_063_lcx1:  {'lcx1': 100.0}
    1494_064_lad:  {'p': 100.0, 'm': 5.838720404902298, 'd': 36.68745656276199}
    1494_064_lcx1:  {'lcx1': 100.0}
    1494_065_lad:  {'p': 100.0, 'm': 16.431222461933615, 'd': 23.264151777165765}
    1494_065_lcx1:  {'lcx1': 100.0}
1523:
    1523_36_lad:  {'p': 63.72286203027373, 'm': 3.7958911881607826, 'd': 65.97340642387115}
    1523_37_lad:  {'p': 30.21880909878204, 'm': 5.164243506763766, 'd': 76.4568854577621}
    1523_38_lad:  {'p': 71.31025846061665, 'm': 10.805841055725828, 'd': 81.61658719446596}
    1523_39_lad:  {'p': 27.176104282734737, 'm': 5.436709156674602, 'd': 72.80871501537123}
    1523_40_lad:  {'p': 45.23986821915314, 'm': 6.634040524184092, 'd': 13.71969519410512}
    1523_41_lad:  {'p': 96.12550494351385, 'm': 5.8827741109146015, 'd': 82.89850748216298}
    1523_42_lad:  {'p': 91.76944083988009, 'm': 16.477131040712866, 'd': 66.87893248888778}
    1523_43_lad:  {'p': 30.910903789178967, 'm': 4.945915765262754, 'd': 100.0}
    1523_44_lad:  {'p': 47.89664415964406, 'd': 82.80810409878893}
    1523_45_lad:  {'p': 27.132863162838206, 'm': 6.626531890416132, 'd': 64.98018008117232}
    1523_46_lad:  {'p': 72.99883228482966, 'm': 9.634903161027363, 'd': 70.42057086291634}
    1523_47_lad:  {'p': 85.94541992379214, 'm': 10.078830819979123, 'd': 60.0866429000378}
    1523_48_lad:  {'p': 100.0, 'm': 16.52386348679459, 'd': 52.48078823439973}
    1523_49_lad:  {'p': 31.128317402680405, 'm': 11.598106849337164, 'd': 59.146594457668144}
    1523_50_lad:  {'p': 78.18668498431649, 'm': 30.89003998453944, 'd': 72.27127220841588}
    1523_51_lad:  {'p': 41.54859660877667, 'm': 20.093899543799854, 'd': 38.73185698457774}
"""


