import os
import pathlib
import pandas as pd
import shutil
import torch
import time
import datetime
from glob import glob
from torchinfo import summary
from torchvision.models import vit_b_32, ViT_B_32_Weights
pd.options.mode.chained_assignment = None

def timestamp_to_seconds(timestamp:str):
    struct = time.strptime(timestamp.split(',')[0],'%H:%M:%S')
    return int(datetime.timedelta(hours=struct.tm_hour,
                              minutes=struct.tm_min,
                              seconds=struct.tm_sec).total_seconds())

workers = os.cpu_count()
videos = glob("/home/vislab-001/Jared/SET-A1/**/*.MP4")

for video in videos:
    video_duration = os.popen(f'ffmpeg -i {video} 2>&1 | grep "Duration"').read()
    timestamp = video_duration.partition('.')[0].partition(':')[-1].strip()
    print(timestamp_to_seconds(timestamp))

"""
TODO
videosToClips:
- before splitting videos into clips using ffmpeg, 
    need to also check video length &
    create clips for empty durations where no distracted behavior happens and label them with -1

    get GNU parallel working for FFmpeg using os.get_cpu()

- annotation file should only have video clip file path and label (format the file as csv but delimit path and label by a ' ')

- change model_train script to go back to using train_test_split on video clip data

- set up PySlowFast with a Model and create config file to use for training (look at repos for examples)

- create post-processing scripts to handle inference and temporal action localization output/accuracy

- add data augmentation (color/flip images horizontally to add more data)
- fixed crop for each camera view (rear, front, side) to only include driver area 
    OR 
    use action detection model to create BBoxes for var crop

- experiment with different models for action classification

- incorporate visual prompting or experiment with other action recognition aspects
"""
