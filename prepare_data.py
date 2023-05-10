import cv2 # capturing videos
import math
import pandas as pd
import keras.utils as image # preprocessing the images
import numpy as np 
from skimage.transform import resize # resizing images
from glob import glob
from tqdm import tqdm
import pathlib

# PyTorch Modules
import torch
from torch.utils.data import Dataset, DataLoader

class UCF101_Dataset(Dataset):
    def __init__(self, images, labels):
      self.images = images
      self.labels = labels
    
    def __len__(self):
      "Returns the total number of samples."
      return len(self.images)
    
    # Returns tuple w/ img tensor and class: (torch.Tensor, int)
    def __getitem__(self, index: int):
      "Returns one sample of data, data and label (X, y)."
      return self.images[index], self.labels[index] # return data, label (X, y)
    

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
        filename = image_filepaths[i].split('/')[1]
        train_imgnames.append(filename)
        train_classes.append(filename.split('_')[1])

    train_data = pd.DataFrame()
    train_data['image'] = train_imgnames
    train_data['class'] = train_classes

    train_data.to_csv(save_dir + "/" + annotation_name, header=True, index=False)

# load saved csv and read images into array as indicated by csv filenames
def loadData(csv_annotation_file, img_dir):
    train = pd.read_csv(csv_annotation_file)

    train_images = []

    # create array of image (frame) data to feed into model as x
    for i in tqdm(range(train.shape[0])):
        img = image.load_img(img_dir + "/" + train['image'][i], target_size=(224,224,3))

        img = image.img_to_array(img)
        img /= 255 # normalize pixel values

        train_images.append(img)

    X = np.array(train_images)
    y = train['class']
    return X,y







