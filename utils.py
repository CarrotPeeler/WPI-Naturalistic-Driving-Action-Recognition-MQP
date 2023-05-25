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

# workers = os.cpu_count()
# videos = glob("/home/vislab-001/Jared/SET-A1/**/*.MP4")

# for video in videos:
#     video_duration = os.popen(f'ffmpeg -i {video} 2>&1 | grep "Duration"').read()
#     timestamp = video_duration.partition('.')[0].partition(':')[-1].strip()
#     print(timestamp_to_seconds(timestamp))


