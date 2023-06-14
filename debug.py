import os
import torch
import pandas as pd
import numpy as np
import cv2
import decord
from PIL import Image
from glob import glob
from scipy import stats

# Always run the start method inside this if-statement
if __name__ == '__main__':  

    # df = pd.read_csv(os.getcwd() + "/slowfast/train.csv", delimiter=" ", names=["path", "class"])
    # print(df.pivot_table(index = ["class"], aggfunc = "size"))
    pass