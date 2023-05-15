import cv2 # capturing videos
import pandas as pd
import numpy as np 
from skimage.transform import resize # resizing images
from glob import glob
from tqdm import tqdm
import pathlib
from PIL import Image
import torch
import os

# PyTorch Modules
from torch.utils.data import Dataset

class UCF101_Dataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.img_dir = img_dir
        self.imgs = dataframe['image']
        self.labels = pd.get_dummies(dataframe['class'], dtype=float)
        self.transform = transform

    def load_image(self, index: int):
        "Opens an image via a path and returns it."
        image_path = self.img_dir + "/" + self.imgs[index]
        return Image.open(image_path) 

    def num_classes(self):
        return self.labels.shape[1]

    def __len__(self):
        "Returns the total number of samples."
        return len(self.imgs)
  
    # Returns tuple w/ img tensor and class: (torch.Tensor, int)
    def __getitem__(self, index: int):
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        label_arr = self.labels.loc[[index]].values # retrieve specified row from dataframe -> convert to numpy array
        label = torch.tensor(label_arr).squeeze() # convert numpy array to tensor -> squeeze to remove extra dim

        # perform transform on image if specified
        if self.transform:
            return self.transform(img), label
        else:
            return img, label


# given text file with class indices and corresponding name/activity, create dictionary for key-val pairs
# pass delimiter parameter that separates key-val pairs in text file
def getClassNamesDict(class_names_txt_path, delimiter):
    d = dict()
    with open(class_names_txt_path) as txt:
        for line in txt:
            (key, val) = line.strip().split(delimiter)
            d[key] = val
    return d


# Break down videos into frames, save frames to selected directory
"""
NOTE: For this function to work, annotation files must be stored as a csv in the same dir as the videos
      This allows the data dir to have multiple subdirs, each with a set of videos and an associated annotation file

video_dir: str
    path to directory where videos stored; 

frame_dir: str
    path to directory where frames will be saved

video_extension: str
    video extension type (.avi, .MP4, etc.) -- include the '.' char and beware of cases!
    
truncate_size: int
    number of frames each video will be truncated to
"""
def videosToFrames(video_dir, frame_dir, video_extension, truncate_size):
    csv_filepaths = glob(video_dir + "/**/*.csv", recursive=True) # search for all .csv files (each dir. of videos should only have ONE)

    for i in tqdm(range(len(csv_filepaths))): # for each csv file corresponding to a group of videos, split each video into frames
        annotation_df = pd.read_csv(csv_filepaths[i])
        videos = glob(csv_filepaths[i].rpartition('/')[0] + "/*" + video_extension)

        for j in range(len(videos)):
            count = 0

            capture = cv2.VideoCapture(videos[j])
            num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) # get the total number of frames in the video
            frame_rate = int(capture.get(cv2.CAP_PROP_FPS)) # get the frames per second

            # select truncate_size num of frames from each video, evenly spaced out
            selected_frames = np.linspace(start=0, stop=num_frames-1, num=truncate_size, dtype=np.int16) 

            while(capture.isOpened()):
                frameId = capture.get(1) # curr frame num
                imageExists, frameImg = capture.read()

                if(imageExists == False):
                    break

                if(frameId in selected_frames): # evenly samples frames at truncate_size interval 
                    save_location = frame_dir + "/" +  vid_file_name + f'_frame{count}.jpg'
                    count += 1
                    cv2.imwrite(save_location, frameImg)
        
        capture.release()

# create csv file to store video frame names and their labels/classes
"""
img_dir: directory where frames are stored
save_dir: where to save annotation
annotation_name: name to save annotation under
"""
def get_frames_annotation(img_dir, save_dir, annotation_name):
    image_filepaths = glob(img_dir + "/*.jpg")

    train_imgnames = []
    train_classes = []

    for i in tqdm(range(len(image_filepaths))):
        filename = image_filepaths[i].split('/')[-1]
        train_imgnames.append(filename)
        train_classes.append(filename.split('_')[1])

    train_data = pd.DataFrame()
    train_data['image'] = train_imgnames
    train_data['class'] = train_classes

    train_data.to_csv(save_dir + "/" + annotation_name, header=True, index=False)







