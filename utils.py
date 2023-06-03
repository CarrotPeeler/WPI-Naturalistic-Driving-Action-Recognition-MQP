import os
import subprocess
import sys
import torch
import mmcv
import pandas as pd
from glob import glob
from typing import List

def get_video_ids_dict(path_to_csv):
    video_ids_dict = dict()
    df = pd.read_csv(path_to_csv)

    for idx, row in df.iterrows():
        key = int(row['video_id'])
        val = row[df.columns[1:4]].to_list()
        video_ids_dict[int(key)] = val

    return video_ids_dict


A2_data_path = "/home/vislab-001/Jared/SET-A2"
video_paths = glob(A2_data_path + "/**/*.MP4")

dic = get_video_ids_dict(os.getcwd() + "/video_ids.csv")
    
video_name = video_paths[0].rpartition('/')
print(video_name)
# video_id =  {i for i in video_ids_dict if video_name in video_ids_dict[i]}