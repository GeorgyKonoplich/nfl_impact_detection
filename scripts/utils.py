
import sys
sys.path.insert(0, "timm-efficientdet-pytorch")
sys.path.insert(0, "omegaconf")

import torch
import os
import random
import cv2
import numpy as np

import pandas as pd
from tqdm import tqdm

SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

def mk_images(video_name, video_labels, video_dir, out_dir, only_with_impact=True):
    video_path=f"{video_dir}/{video_name}"
    video_name = os.path.basename(video_path)
    vidcap = cv2.VideoCapture(video_path)
    if only_with_impact:
        boxes_all = video_labels.query("video == @video_name")
        print(video_path, boxes_all[boxes_all.impact > 1.0].shape[0])
    else:
        print(video_path)
    frame = 0
    while True:
        it_worked, img = vidcap.read()
        if not it_worked:
            break
        frame += 1
        if only_with_impact:
            boxes = video_labels.query("video == @video_name and frame == @frame")
            boxes_with_impact = boxes[boxes.impact > 1.0]
            if boxes_with_impact.shape[0] == 0:
                continue
        img_name = f"{video_name}_frame{frame}"
        image_path = f'{out_dir}/{video_name}'.replace('.mp4',f'_{frame}.png')
        _ = cv2.imwrite(image_path, img)

video_labels = pd.read_csv('../data/train_labels.csv').fillna(0)
video_labels_with_impact = video_labels[video_labels['impact'] > 0]
for row in tqdm(video_labels_with_impact[['video','frame','label']].values):
    frames = np.array([-4,-3,-2,-1,1,2,3,4])+row[1]
    video_labels.loc[(video_labels['video'] == row[0])
                                 & (video_labels['frame'].isin(frames))
                                 & (video_labels['label'] == row[2]), 'impact'] = 1
video_labels['image_name'] = video_labels['video'].str.replace('.mp4', '') + '_' + video_labels['frame'].astype(str) + '.png'
video_labels = video_labels[video_labels.groupby('image_name')['impact'].transform("sum") > 0].reset_index(drop=True)
video_labels['impact'] = video_labels['impact'].astype(int)+1
video_labels['x'] = video_labels['left']
video_labels['y'] = video_labels['top']
video_labels['w'] = video_labels['width']
video_labels['h'] = video_labels['height']
video_labels.head()
video_labels.to_csv('video_labels.csv')

uniq_video = video_labels.video.unique()
video_dir = '../data/train'
out_dir = '../data/train_images_only_impact'

for video_name in uniq_video:
    mk_images(video_name, video_labels, video_dir, out_dir)

