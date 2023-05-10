import cv2 # capturing videos
import math
import pandas as pd
import numpy as np 
from skimage.transform import resize # resizing images
from glob import glob
from tqdm import tqdm
import pathlib
from PIL import Image

# PyTorch Modules
from torch.utils.data import Dataset

class UCF101_Dataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
      self.img_dir = img_dir
      self.imgs = dataframe['image']
      self.labels = pd.get_dummies(dataframe['class'])

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
      label = self.labels[index]

      # perform transform on image if specified
      if self.transform:
        return self.transform(img), label
      else:
        return img, label
    

# returns a dataframe that stores the video names and their labels, given a raw txt file
def createDataFrame(txt_file_location):
  file = open(txt_file_location, 'r')
  video_names = file.read().split('\n')
  dataframe = pd.DataFrame()
  dataframe['video_name'] = video_names

  labels =[]
  for i in tqdm(range(dataframe.shape[0])):
      labels.append(dataframe['video_name'][i].split('/')[0])

  dataframe['label'] = labels

  return dataframe

def filterDataFrame(dataframe, filterList):
  for index, row in tqdm(dataframe.iterrows()):
    if not any([keyword in row['label'].lower() for keyword in filterList]):
      dataframe.drop(index, inplace=True)
  dataframe.reset_index(drop=True, inplace=True)

# Break down videos into frames, save frames to selected directory
def videoToFrames(dataframe, video_dir, frame_dir):
    for i in tqdm(range(dataframe.shape[0])):
        count = 0
        vid_file_name = dataframe['video_name'][i].split(' ')[0].split('/')[1]

        capture = cv2.VideoCapture(video_dir + "/" + vid_file_name)
        frameRate = capture.get(5) # get frame rate property

        while(capture.isOpened()):
            frameId = capture.get(1) # curr frame num
            hasImage, frameImg = capture.read()

            if(hasImage == False):
                break

            if(frameId % math.floor(frameRate) == 0): # save only 1 frame per sec (per frame rate)
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
def create_annotation(img_dir, save_dir, annotation_name):
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







