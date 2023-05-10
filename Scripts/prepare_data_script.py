from prepare_data import *
import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n\nDevice: {device}\n\n")

# Filter the UCF101 dataset by action types
filter = ['apply','blowdry','haircut','brushing','typing']

# Create train and test dataframes
train = createDataFrame('/home/vislab-001/Jared/ucfTrainTestlist/trainlist01.txt')
test = createDataFrame('/home/vislab-001/Jared/ucfTrainTestlist/testlist01.txt')

filterDataFrame(train,filter)
filterDataFrame(test,filter)

train_1_dir = os.getcwd() + "/train_1"

# Split train videos into frames
videoToFrames(train, "/home/vislab-001/Jared/ucf101", train_1_dir)

# Create annotation file (lists all frame filenames and their labels)
create_annotation(train_1_dir, train_1_dir, "train_1_annotation.csv")