import pandas as pd
import os
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold

# suppress pandas chain assignment warnings
pd.options.mode.chained_assignment = None

"""
Given text file with class indices and corresponding name/activity, create dictionary for key-val pairs
"""
def getClassNamesDict(class_names_txt_path):
    d = dict()
    with open(class_names_txt_path) as txt:
        for line in txt:
            (key, val) = line.strip().split(",")
            d[int(key)] = val
    return d



"""
Converts timestamp in string format "hh:mm:ss" to total seconds
Returns an integer representing total seconds
"""
def timestamp_to_seconds(timestamp:str):
    hours, mins, secs = timestamp.split(':')
    return int(hours)*3600 + int(mins)*60 + int(secs)



"""
Extract video duration in seconds 
"""
def extract_video_duration_seconds(video_filepath:str):
    video_metadata = os.popen(f'ffmpeg -i {video_filepath} 2>&1 | grep "Duration"').read()
    timestamp = video_metadata.partition('.')[0].partition(':')[-1].strip()
    return timestamp_to_seconds(timestamp)



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

    if(sum == video_duration):
        return True, corrected_video_action_labels
    else:
        return False, corrected_video_action_labels



"""
Breaks down videos into clips, each having 1 action in them

Clips saved to given dir. and creates an annotation csv with clip file names and their labels

NOTE: For this function to work, annotation files must be stored as a csv in the same dir as the videos
      This allows the data dir to have multiple subdirs, each with a set of videos and an associated annotation file

video_dir: path to directory where videos stored
clip_dir: path to directory where clips will be saved
video_extension: video extension type (.avi, .MP4, etc.) -- include the '.' char and beware of cases!
annotation_filename: name to save annotation under (you must supply the extension type)
re_encode: true for re-encoding, false otherwise
clip_resolution: i.e., -2:540, 720x540, etc.; only applied when re_encode = True
encode_speed: "default" for preset speed; else, use any ffmpeg speed param

Returns True if operation suceeded; else, False if it failed
"""
def videosToClips(video_dir: str, clip_dir: str, annotation_filename: str, video_extension: str, re_encode:bool, clip_resolution:str, encode_speed:str):
    csv_filepaths = glob(video_dir + "/**/*.csv", recursive=True) # search for all .csv files (each dir. of videos should only have ONE)
    clip_filepaths = [] # stores image (frame) names
    classes = [] # stores class labels for each frame

    for i in tqdm(range(len(csv_filepaths))): # for each csv file corresponding to a group of videos, split each video into clips
        annotation_df = pd.read_csv(csv_filepaths[i])
        videos = glob(csv_filepaths[i].rpartition('/')[0] + "/*" + video_extension)

        for video in videos: # for each video, parse out its individual csv data from the original csv for the group of videos 
            video_filename = video.rpartition('/')[-1].partition('.')[0] 
            parsed_video_df = parse_data_from_csv(video, annotation_df)
            video_duration = extract_video_duration_seconds(video) # video duration in total seconds elapsed

            # extract start and end timestamps from df and convert each to seconds
            start_times = list(map(timestamp_to_seconds, parsed_video_df['Start Time'].to_list()))
            end_times = list(map(timestamp_to_seconds, parsed_video_df['End Time'].to_list()))
            labels = list(map(lambda str:int(str.rpartition(' ')[-1]), parsed_video_df['Label (Primary)'].to_list()))

            retval, video_action_labels = fill_unlabeled_video_segments(start_times=start_times, 
                                                                end_times=end_times, 
                                                                labels=labels, 
                                                                video_duration=video_duration)
            
            # if creating labels for unlabeled segments failed, halt process
            if(retval == False): 
                print(parsed_video_df)
                print(f"Failed to auto-generate labels for unlabeled video segments for:\n{video}\nResult:{video_action_labels}")
                return False
                
            # write ffmpeg commands to bash script for parallel execution & add labels and file paths to lists for annotation 
            with open(clip_dir + "/ffmpeg_commands.sh", 'a+') as f:

                for action_tuple in video_action_labels: # for each action in the video, extract video clip of action based on timestamps

                    # extract only the portion of the video between start_time and end_time
                    clip_filepath = clip_dir + f"/{video_filename}" + f"-start{action_tuple[0]}" + f"-end{action_tuple[1]}" + ".MP4"

                    if(encode_speed == "default"): 
                        preset = ""
                    else:
                        preset = "-preset " + encode_speed + " "
                    
                    # # no re-encoding (typically much faster than with re-encoding)
                    if(re_encode == False):
                        f.writelines(f"ffmpeg -loglevel quiet -y -i {video} -ss {action_tuple[0]} -to {action_tuple[1]} -c:v copy {clip_filepath}\n")
                    else:       
                        f.writelines(f"ffmpeg -loglevel quiet -y -i {video} -vf scale={clip_resolution} -ss {action_tuple[0]} -to {action_tuple[1]} -c:v libx264 {preset}{clip_filepath}\n")
                    
                    clip_filepaths.append(clip_filepath)
                    classes.append(action_tuple[2])
    
    # parallelize ffmpeg commands
    os.system(f"parallel --eta < {clip_dir}/ffmpeg_commands.sh")

    # create annotation csv to store clip file paths and their labels
    data = pd.DataFrame()
    data['clip'] = clip_filepaths
    data['class'] = classes  
    data.to_csv(clip_dir + "/" + annotation_filename, sep=" ", header=False, index=False)
    return True



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
    annotation_filename = "annotation.csv"

    # truncate each train video into frames (truncate_size = num of frames per video)
    if(videosToClips(video_dir="/home/vislab-001/Jared/SET-A1", 
                  clip_dir=clips_savepath, 
                  video_extension=".MP4", 
                  annotation_filename=annotation_filename,
                  re_encode=True,
                  encode_speed = "ultrafast",
                  clip_resolution="512:512")):

        print("All videos have been successfully processed into clips. Creating annotation split...")

        df = pd.read_csv(clips_savepath + "/" + annotation_filename, sep=" ", names=["clip", "class"])

        # split data into train and test sets via grouping by video name
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state = 42)
        split = splitter.split(X=df['clip'], y=df['class'], groups=df['clip'].str.rpartition('/')[2].str.partition('-')[0])
        train_indexes, test_indexes = next(split)

        train_df = df.iloc[train_indexes]
        test_df = df.iloc[test_indexes]

        train_df.to_csv(clips_savepath + "/A1_train.csv", sep=" ", header=False, index=False)
        test_df.to_csv(clips_savepath + "/A1_test.csv", sep=" ", header=False, index=False)

    else:
        print("Video processing was unsuccessful.")