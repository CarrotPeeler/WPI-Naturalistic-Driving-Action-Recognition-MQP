import os
import pandas as pd
import time
import datetime
from sklearn.model_selection import StratifiedGroupKFold
from glob import glob
from torchinfo import summary
pd.options.mode.chained_assignment = None

def timestamp_to_seconds(timestamp:str):
    struct = time.strptime(timestamp.split(',')[0],'%H:%M:%S')
    return int(datetime.timedelta(hours=struct.tm_hour,
                              minutes=struct.tm_min,
                              seconds=struct.tm_sec).total_seconds())

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

Returns a list of tuples (start_time, end_time, class_id_label) that classify every segment of the video under an action id
or returns None if operation failed
"""
def fill_unlabeled_video_segments(start_times:list, end_times:list, labels:list, video_duration:int):
    
    # create a list of tuples (start_time, end_time, class_id) for the video
    video_action_labels = [(start_times[i], end_times[i], labels[i]) for i in range(0, len(labels))]

    corrected_video_action_labels = [] # includes labels and timestamps for unlabeled non-distracted behavior segments in the video

    # add labels and timestamps for unlabeled non-distracted behavior segments of the video (categorized as class 0 - normal driving)
    for i, tuple in enumerate(video_action_labels):
        # first action segment does not occur at start of video => add normal driving tuple before this action tuple
        if(i == 0 and tuple[0] > 0):
            corrected_video_action_labels.append((0, tuple[0], 0)) # class id of 0 => normal driving
            corrected_video_action_labels.append(tuple)

        # last action segment does not last till end of video duration => add normal driving tuple after this distracted action tuple
        elif(i == len(video_action_labels)-1 and tuple[1] < video_duration):
            corrected_video_action_labels.append(tuple)
            corrected_video_action_labels.append((tuple[1], video_duration, 0))

        # time gap between when this action occurs and the next labeled action = > add action tuple for normal driving after this tuple
        if(i+1 < len(video_action_labels) and video_action_labels[i+1][0] - tuple[1] > 0):
            if(i != 0): # prevent first action tuple from being duplicated if first 'if' statement True
                corrected_video_action_labels.append(tuple)
            corrected_video_action_labels.append((tuple[1], video_action_labels[i+1][0], 0))

    # check that there are timestamps and labels for every segment of the video
    sum = corrected_video_action_labels[-1][1] # == video duration
    for i, tuple in enumerate(corrected_video_action_labels):
        if(i+1 < len(corrected_video_action_labels)):
            sum += (corrected_video_action_labels[i+1][0] - tuple[1])

    if(sum == video_duration):
        return corrected_video_action_labels
    else:
        return None


if __name__ == '__main__':

    workers = os.cpu_count()
    csv_path = glob("/home/vislab-001/Jared/SET-A1/**/*.csv")[0]
    video_path = glob(csv_path.rpartition("/")[0]+"/*.MP4")[0]

    video_metadata = os.popen(f'ffmpeg -i {video_path} 2>&1 | grep "Duration"').read()
    timestamp = video_metadata.partition('.')[0].partition(':')[-1].strip()
    video_duration = timestamp_to_seconds(timestamp)
    
    df = pd.read_csv(csv_path)
    parsed_df = parse_data_from_csv(video_filepath=video_path, annotation_dataframe=df)
    start_times = list(map(timestamp_to_seconds, df['Start Time'].to_list()))
    end_times = list(map(timestamp_to_seconds, df['End Time'].to_list()))
    labels = list(map(lambda str:int(str.rpartition(' ')[-1]), parsed_df['Label (Primary)'].to_list()))

    print(fill_unlabeled_video_segments(start_times=start_times, end_times=end_times, labels=labels, video_duration=video_duration))


    # df = pd.read_csv("tets.csv", sep=" ", names=["clip", "class"])

    # # split data into train and test sets using the group ids
    # splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state = 42)
    # split = splitter.split(X=df['clip'], y=df['class'], groups=df['clip'].str.partition('-')[0].str.rpartition('/')[2])
    # train_indexes, test_indexes = next(split)

    # train_df = df.iloc[train_indexes]
    # test_df = df.iloc[test_indexes]

    # train_df.to_csv("tets_train", sep=" ", header=False, index=False)
    # test_df.to_csv("tets_test", sep=" ", header=False, index=False)

    