import os
import subprocess
import sys
import torch
import mmcv
import decord
import pandas as pd
import numpy as np
import time
from glob import glob
from typing import List

def temporal_sampling(frames, start_frame_idx, end_frame_idx, num_samples):
    idxs = np.linspace(start_frame_idx, end_frame_idx, num_samples, dtype=np.int16)
    idxs = np.clip(idxs, 0, len(frames) - 1)
    sampled_frames = [frames[i] for i in idxs]
    return sampled_frames

def temporal_sampling_3(frames, start_frame_idx, end_frame_idx, num_samples):
    idxs = torch.linspace(start_frame_idx, end_frame_idx, num_samples, dtype=torch.int16)
    idxs = torch.clamp(idxs, 0, len(frames) - 1).numpy().tolist()
    sampled_frames = [frames[i] for i in idxs]
    return sampled_frames

def temporal_sampling_2(frames, start_frame_idx, end_frame_idx, num_samples):
    idxs = torch.linspace(start_frame_idx, end_frame_idx, num_samples)
    idxs = torch.clamp(idxs, 0, len(frames) - 1).long()
    sampled_frames = torch.index_select(torch.from_numpy(frames), 0, idxs)
    return sampled_frames

# Always run the start method inside this if-statement
if __name__ == '__main__':  

    A2_data_path = "/home/vislab-001/Jared/SET-A2"
    video_paths = glob(A2_data_path + "/**/*.MP4")

    video = decord.VideoReader(video_paths[0], num_threads=os.cpu_count())
    frames = video.get_batch(list(range(len(video)))).asnumpy()

    times1 = 0
    times2 = 0
    times3 = 0
    trials = 20
    for i in range(trials+1):
        # start_time1 = time.time()
        # temporal_sampling(frames, 0, 63, 16)
        # end_time1 = time.time()
        # times1 += (end_time1-start_time1)

        start_time2 = time.time()
        temporal_sampling_2(frames, 0, 63, 16)
        end_time2 = time.time()
        times2 += (end_time2-start_time2)

        # start_time3 = time.time()
        # temporal_sampling_3(frames, 0, 63, 16)
        # end_time3 = time.time()
        # times3 += (end_time3-start_time3)
    times1 /= trials
    times2 /= trials
    times3 /= trials

    print(f"numpy: {times1:.2f} seconds")
    print(f"torch-mod: {times2:.2f} seconds")
    print(f"torch-orig: {times3:.2f} seconds")

    # video = decord.VideoReader(video_paths[0], num_threads=os.cpu_count())
    # frames = video.get_batch(list(range(len(video))))
    # trials = 20
    # times3 = 0
    # for i in range(trials+1):
    #     start_time3 = time.time()
        
    #     end_time3 = time.time()
    #     times3 += (end_time3-start_time3)
    # times3 /= trials

    # print(f"decord: {times3:.2f} seconds")
