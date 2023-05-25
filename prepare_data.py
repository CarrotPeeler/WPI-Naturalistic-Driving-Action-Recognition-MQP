import pandas as pd
from glob import glob
from tqdm import tqdm
import os
import datetime
import time

# suppress pandas chain assignment warnings
pd.options.mode.chained_assignment = None

"""
Given text file with class indices and corresponding name/activity, create dictionary for key-val pairs
pass delimiter parameter that separates key-val pairs in text file
"""
def getClassNamesDict(class_names_txt_path, delimiter):
    d = dict()
    with open(class_names_txt_path) as txt:
        for line in txt:
            (key, val) = line.strip().split(delimiter)
            d[key] = val
    return d


"""
Converts timestamp in string format "hh:mm:ss" to total seconds
Returns an integer representing total seconds
"""
def timestamp_to_seconds(timestamp:str):
    struct = time.strptime(timestamp.split(',')[0],'%H:%M:%S')
    return int(datetime.timedelta(hours=struct.tm_hour,
                              minutes=struct.tm_min,
                              seconds=struct.tm_sec).total_seconds())


"""
Breaks down videos into clips, each having 1 action in them

Clips saved to given dir. and creates an annotation csv with clip file names and their labels

NOTE: For this function to work, annotation files must be stored as a csv in the same dir as the videos
      This allows the data dir to have multiple subdirs, each with a set of videos and an associated annotation file

video_dir: path to directory where videos stored
clip_dir: path to directory where clips will be saved
video_extension: video extension type (.avi, .MP4, etc.) -- include the '.' char and beware of cases!
"""
def videosToClips(video_dir: str, clip_dir: str, video_extension: str):
    csv_filepaths = glob(video_dir + "/**/*.csv", recursive=True) # search for all .csv files (each dir. of videos should only have ONE)
    clip_filenames = [] # stores image (frame) names
    classes = [] # stores class labels for each frame
    video_idxs = [] # indices that indicate which video a frame came from

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
                trimmed_video_filepath = os.getcwd() + "/data" + f"/{video_filename}_" + class_label.replace(" ","") + f"_trim{k}" + ".MP4"
                os.system(f"ffmpeg -loglevel quiet -i {videos[j]} -ss {start_time} -to {end_time} -c:v copy {trimmed_video_filepath}")

                image_filenames.extend(images)
                classes += len(images) * [class_label]
                video_idxs += len(images) * [j]

    # create annotation csv to store image names and their labels
    data = pd.DataFrame()
    data['image'] = image_filenames
    data['class'] = classes   
    data['video_index'] = video_idxs
    data.to_csv(frame_dir + "/annotation.csv", header=True, index=False)

"""
Parses out the data for a single video, given its file path and the entire annotation csv for the user recorded in the video
NOTE: each annotation file has multiple videos, each with their own data
"""
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



if __name__ == '__main__':
    
    clips_savepath = os.getcwd() + "/data"

    # truncate each train video into frames (truncate_size = num of frames per video)
    videosToClips(video_dir="/home/vislab-001/Jared/SET-A1", clip_dir=clips_savepath, video_extension=".MP4")