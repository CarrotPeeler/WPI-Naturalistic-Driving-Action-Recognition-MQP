import mmcv # capturing videos
import pandas as pd
import numpy as np 
from skimage.transform import resize # resizing images
from glob import glob
from tqdm import tqdm
import pathlib
from PIL import Image
import torch
import os
import shutil

# suppress pandas chain assignment warnings
pd.options.mode.chained_assignment = None

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


"""
Breaks down videos into frames, save them to given dir., and creates an annotation csv with frame file names and their labels

NOTE: For this function to work, annotation files must be stored as a csv in the same dir as the videos
      This allows the data dir to have multiple subdirs, each with a set of videos and an associated annotation file

video_dir: str
    path to directory where videos stored; 

frame_dir: str
    path to directory where frames will be saved

num_samples: int
    number of frames to sample for each action video clip

video_extension: str
    video extension type (.avi, .MP4, etc.) -- include the '.' char and beware of cases!
"""
def videosToFrames(video_dir, frame_dir, video_extension, num_samples):
    csv_filepaths = glob(video_dir + "/**/*.csv", recursive=True) # search for all .csv files (each dir. of videos should only have ONE)
    image_filenames = [] # stores image (frame) names
    classes = [] # stores class labels for each frame
    video_idxs = [] # indices that indicate which video a frame came from

    # dump path for trimmed video clips
    dump_path = os.getcwd() + "/trimmed_video_dump"
    pathlib.Path(dump_path).mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(csv_filepaths))): # for each csv file corresponding to a group of videos, split each video into frames
        annotation_df = pd.read_csv(csv_filepaths[i])
        videos = glob(csv_filepaths[i].rpartition('/')[0] + "/*" + video_extension)

        for j in range(len(videos)): # for each video, parse out its individual csv data from the original csv 
            video_filename = videos[j].rpartition('/')[-1].partition('.')[0] 
            parsed_video_data = parse_data_from_csv(videos[j], annotation_df)

            for k in range(len(parsed_video_data)): # for each row in the parsed csv file, extract video clip based on timestamps; split clip into frames
                row_data = parsed_video_data.iloc[[k]] # extract a single row in the csv

                start_time = row_data['Start Time'].to_list()[0]
                end_time = row_data['End Time'].to_list()[0]
                class_label = row_data['Label (Primary)'].to_list()[0] 

                # extract only the portion of the video between start_time and end_time
                trimmed_video_filepath = dump_path + f"/{video_filename}_" + class_label.replace(" ","") + f"_trim{k}" + ".MP4"
                os.system(f"ffmpeg -loglevel quiet -i {videos[j]} -ss {start_time} -to {end_time} -c:v copy {trimmed_video_filepath}")

                images = splitVideoClip(trimmed_video_filepath, frame_dir, num_samples)
                image_filenames.extend(images)
                classes += num_samples * [class_label]
                video_idxs += num_samples * [j]

    # delete trimmed video dump dir.
    shutil.rmtree(dump_path, ignore_errors=True)

    # create annotation csv to store image names and their labels
    data = pd.DataFrame()
    data['image'] = image_filenames
    data['class'] = classes   
    data['video_index'] = video_idxs
    data.to_csv(frame_dir + "/annotation.csv", header=True, index=False)


# returns only the relevant data from the annotation csv file for the video requested
# each annotation file has multiple videos, each with their own data; the goal is to parse data for individual videos\
def parse_data_from_csv(video_filepath, annotation_dataframe):
    df = annotation_dataframe

    video_view = video_filepath.rpartition('/')[-1].partition('_')[0] # retrieve camera view angle
    video_endnum = video_filepath.rpartition('_')[-1].partition('.')[0] # retrieve block appearance number (last number in the file name)

    video_start_rows = df.loc[df["Filename"].notnull()] # create dataframe with only rows having non-null file names
    video_start_rows.reset_index(inplace=True)

    for index, row in video_start_rows.iterrows(): # rename each row; only include the camera view type and block number
        video_start_rows["Filename"][index] = row["Filename"].partition('_')[0] + "_" + row["Filename"].rpartition('_')[-1]

    # with sub-dataframe, retrieve zero-based index for the current video 
    video_index = video_start_rows.index[video_start_rows["Filename"].str.contains(video_view, case=False) &
                                        video_start_rows["Filename"].str.contains(video_endnum, case=False)].to_list()[0]
    
    video_index_orig = video_start_rows.iloc[[video_index]]["index"].to_list()[0] # find the original dataframe index 

    next_video_index_orig = -1

    if video_index + 1 < len(video_start_rows): # if there's data for other videos after this video, grab the index where the next video's data starts
        next_video_index_orig = video_start_rows.iloc[[video_index + 1]]["index"].to_list()[0]
    else:
        next_video_index_orig = len(df) - 1 # otherwise, this video's data is last in the csv, simply set the 

    parsed_video_data = df.iloc[video_index_orig:next_video_index_orig] # create a sub-dataframe of the original with only this video's data

    return parsed_video_data


"""
Splits a video clip containing a single action/activity into frames; saves frames to 'frame_dir' location

video_filepath: str
    path to where video file is stored; 

frame_dir: str
    path to directory where frames will be saved

num_samples: int
    number of frames to sample from each video

Returns tuple containing a list of image names and a list of classes associated with them
"""
def splitVideoClip(video_filepath, class_label, frame_dir, num_samples=None):
    videoclip_frames = mmcv.VideoReader(video_filepath)
    
    video_filename = video_filepath.rpartition('/')[-1].rpartition('.')[0]
    num_frames = videoclip_frames.frame_cnt

    # if no sample number given, default to sample every frame
    if(num_samples == None): 
        num_samples = num_frames

    # evenly sample frames from a video following the given 
    sampled_frame_idxs = torch.linspace(start=0, end=num_frames-1, steps=num_samples, dtype=torch.int16) 

    image_filenames = []

    count = 0
    for i, frame_img in enumerate(videoclip_frames):
        if(i in sampled_frame_idxs):
            frame_filename = video_filename + f'_frame{count}.jpg'

            # append image
            image_filenames.append(frame_filename) 

            # Save the image to specified directory
            save_location = frame_dir + "/" +  frame_filename
            mmcv.imwrite(frame_img, save_location)
            count += 1

    return image_filenames









