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
    vid_file = glob("/home/vislab-001/Jared/SET-A2/**/*.MP4")[0]
    vid = decord.VideoReader(vid_file, num_threads=20)
   
    # Below VideoWriter object will create
    # a frame of above defined The output 
    # is stored in 'filename.avi' file.
    clip = cv2.VideoWriter(os.getcwd() + '/clip_20s.mp4', 
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            30, (512,512))
        
    frames = vid.get_batch(list(range(600))).asnumpy()

    for frame in frames:
        new_frame = cv2.resize(frame, (512,512))
        clip.write(new_frame)

    clip.release()
    
    print("The video was successfully saved")

#  # split data into train and val sets
#         splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state = 42)
#         split = splitter.split(X=df['clip'], y=df['class'])
#         train_indexes, test_indexes = next(split)

#         train_df = df.iloc[train_indexes]
#         test_df = df.iloc[test_indexes]

#         train_df.to_csv(os.getcwd() + "/slowfast/train.csv", sep=" ", header=False, index=False)
#         test_df.to_csv(os.getcwd() + "/slowfast/val.csv", sep=" ", header=False, index=False)