# creates dictionary of each track with all the bounding boxes per frame in each track and saves it to args.savePath/trackid.json

import os
import pandas as pd 
import numpy as np
from glob import glob
from tqdm import tqdm
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--trackPath', type = str, default = '/users/acp21jrc/audio-visual/active-speaker-detection/active_speaker/TalkNet_ASD/Ego4d_TalkNet_ASD/data/track_results')
parser.add_argument('--annotPath', type = str, default = '/users/acp21jrc/audio-visual/active-speaker-detection/active_speaker/TalkNet_ASD/Ego4d_TalkNet_ASD/data/infer/csv') # '/users/acp21jrc/audio-visual/active-speaker-detection/active_speaker/TalkNet_ASD/Ego4d_TalkNet_ASD/data/ego4d/csv'
parser.add_argument('--evalDataType', type = str, default = 'test') # val
parser.add_argument('--savePath', type = str, default = '/users/acp21jrc/audio-visual/active-speaker-detection/active_speaker/TalkNet_ASD/Ego4d_TalkNet_ASD/data/bboxes_per_track')
args = parser.parse_args()

# open v.txt
with open('/users/acp21jrc/audio-visual/active-speaker-detection/active_speaker/TalkNet_ASD/Ego4d_TalkNet_ASD/v.txt') as f:
    v = f.readlines()
v = [x.strip().split('.')[0] for x in v]

all_files = [os.path.join(args.trackPath, f'{vid}.txt') for vid in v]

# check if each fname in all_files exists
for fname in all_files:
    if not os.path.exists(fname):
        print(f'{fname} does not exist')
        break

# open each file in all_files and collate into a dictionary of shape {vid: [{frame: pid: x1: y1: x2: y2: activity:}]}
dfs = {}
for fname in tqdm(all_files):
    vid = os.path.basename(fname).split('.')[0]
    df = pd.read_csv(fname, sep=' ', header=None, names=['frame', 'pid', 'x1', 'y1', 'x2', 'y2', 'activity'])
    df = df.groupby('frame').apply(lambda x: x.to_dict('records')).to_dict()
    dfs[vid] = df
    #break
tracks = pd.read_csv(os.path.join(args.annotPath, f'active_speaker_{args.evalDataType}.csv'), sep='\t', header=None, names=['trackid', 'duration', 'frame_rate', 'activities', 'start_frame'])

# create dictionary of shape {trackid: [start_frame, strat_frame + duration]}
track_dict = {}
for index, row in tracks.iterrows():
    track_dict[row['trackid']] = [row['start_frame'], row['start_frame'] + row['duration']]
    
# iterate through track_dict
bboxes_dict = {}
for trackid in tqdm(track_dict):
    vid = trackid.split(':')[0]
    frame_dict = {}
    #print(trackid)
    for frame in range(track_dict[trackid][0], track_dict[trackid][1]):
        frame_dict[frame] = []#
        if frame not in dfs[vid]:
            continue
        for pid in dfs[vid][frame]:
            #print(pid)
            frame_dict[frame].append({'x1': pid['x1'], 'y1': pid['y1'], 'x2': pid['x2'], 'y2': pid['y2']})
        #break
    bboxes_dict[trackid] = frame_dict
    #break
# iterate through bboxes_dict and save each trackid as a json file
for trackid in tqdm(bboxes_dict):
    saveFile = os.path.join(args.savePath, f'{trackid}.json')
    with open(saveFile, 'w') as f:
        json.dump(bboxes_dict[trackid], f)    
    
# # save bbox_dict as a json file
# import json
# with open('bboxes_val.json', 'w') as f:
#     json.dump(bboxes_dict, f)

