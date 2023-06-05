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

# Always run the start method inside this if-statement
if __name__ == '__main__':  

    A2_data_path = "/home/vislab-001/Jared/SET-A2"
    video_paths = glob(A2_data_path + "/**/*.MP4")

    with open(os.getcwd() + "/slowfast/test.csv", "a+") as f:
        for video_path in video_paths:
            f.writelines(f"{video_path} 0\n")