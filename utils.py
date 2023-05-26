import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
from glob import glob
from torchinfo import summary
pd.options.mode.chained_assignment = None

def timestamp_to_seconds(timestamp:str):
    hours, mins, secs = timestamp.split(':')
    return int(hours)*3600 + int(mins)*60 + int(secs)

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
If a video is not fully labeled (i.e., not every segment of the video has an action label, then fill in empty segments with labels)

start_times: list of start times when action labels occur for a video segment (each time must be in integer seconds, not hh:mm:ss format)
end_times: list of end times when action labels end for a video segment (each time must be in integer seconds, not hh:mm:ss format)
labels: class ids that classify every segment of the video (must be an integer 0-15)
video_duration: duration of the video in seconds

Returns a list of tuples (start_time, end_time, class_id_label) and True if operation was successful
Else, returns the list and False is operation failed
"""
def fill_unlabeled_video_segments(start_times:list, end_times:list, labels:list, video_duration:int):
    
    # create a list of tuples (start_time, end_time, class_id) for the video
    video_action_labels = [(start_times[i], end_times[i], labels[i]) for i in range(0, len(labels))]
    
    # sort list of tuples by start time
    video_action_labels.sort(key=lambda tuple: tuple[0]) 

    corrected_video_action_labels = [] # includes labels and timestamps for unlabeled non-distracted behavior segments in the video

    # add labels and timestamps for unlabeled non-distracted behavior segments of the video (categorized as class 0 - normal driving)
    curr_start_time = 0 
    for i, action_tuple in enumerate(video_action_labels):

        if(action_tuple[0] == curr_start_time):
            corrected_video_action_labels.append(action_tuple)

        else:
            corrected_video_action_labels.append((curr_start_time, action_tuple[0], 0))
            corrected_video_action_labels.append(action_tuple)

        if(i == len(video_action_labels)-1 and action_tuple[1] < video_duration):
            corrected_video_action_labels.append((action_tuple[1], video_duration, 0))

        curr_start_time = action_tuple[1]

    # check if last tuple's end_time has a 1-sec discrepancy with the video duration (at least a few videos do)
    if(abs(corrected_video_action_labels[-1][1] - video_duration) == 1):
        action_tuple = corrected_video_action_labels[-1]
        new_tuple = (action_tuple[0], video_duration, action_tuple[2])
        del corrected_video_action_labels[-1]
        corrected_video_action_labels.append(new_tuple)
        
        
    # check that action tuples are sequenced in the list correctly based on their timestamps
    sum = corrected_video_action_labels[-1][1] # == video duration
    for i, tuple in enumerate(corrected_video_action_labels):
        if(i+1 < len(corrected_video_action_labels)):
            sum += (corrected_video_action_labels[i+1][0] - tuple[1])

    # videos might have a 1 sec diff. with their annotations (only for 86952_11)
    if(sum == video_duration):
        return True, corrected_video_action_labels
    else:
        return False, corrected_video_action_labels


if __name__ == '__main__':

    csvs = glob("/home/vislab-001/Jared/SET-A1/**/*.csv", recursive=True)

    for i in tqdm(range(len(csvs))):
        videos = glob(csvs[i].rpartition('/')[0] + "**/*.MP4", recursive=True)

        for video in videos:
            video_metadata = os.popen(f'ffmpeg -i {video} 2>&1 | grep "Duration"').read()
            timestamp = video_metadata.partition('.')[0].partition(':')[-1].strip()
            video_duration = timestamp_to_seconds(timestamp)

            df = pd.read_csv(csvs[i])
            parsed_df = parse_data_from_csv(video_filepath=video, annotation_dataframe=df)
            start_times = list(map(timestamp_to_seconds, parsed_df['Start Time'].to_list()))
            end_times = list(map(timestamp_to_seconds, parsed_df['End Time'].to_list()))
            labels = list(map(lambda str:int(str.rpartition(' ')[-1]), parsed_df['Label (Primary)'].to_list()))

            _, actions = fill_unlabeled_video_segments(start_times=start_times, end_times=end_times, labels=labels, video_duration=video_duration)
            if(_ == False): 
                print(f"{parsed_df}\n\nVideo: {video}\n\n{video_metadata}\n\n{video_duration}\n\nCorrected: {actions}\n\n")

    # i = 10
    # video = glob(csvs[i].rpartition('/')[0] + "**/*.MP4", recursive=True)[0]
    # video_metadata = os.popen(f'ffmpeg -i {video} 2>&1 | grep "Duration"').read()
    # timestamp = video_metadata.partition('.')[0].partition(':')[-1].strip()
    # video_duration = timestamp_to_seconds(timestamp)
    # print(f"Video Duration: {video_duration} seconds")

    # df = pd.read_csv(csvs[i])
    # parsed_df = parse_data_from_csv(video_filepath=video, annotation_dataframe=df)
    # start_times = list(map(timestamp_to_seconds, df['Start Time'].to_list()))
    # end_times = list(map(timestamp_to_seconds, df['End Time'].to_list()))
    # labels = list(map(lambda str:int(str.rpartition(' ')[-1]), parsed_df['Label (Primary)'].to_list()))

    # _, actions, actions_orig = fill_unlabeled_video_segments(start_times=start_times, end_times=end_times, labels=labels, video_duration=video_duration)
    # if(_ == False): 
    #     print(f"Before: {actions_orig}\n\nCorrected: {actions}\n\n")