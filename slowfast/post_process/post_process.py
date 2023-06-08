import os
import pandas as pd
from glob import glob
from prepare_data import timestamp_to_seconds

"""
Returns the length of the shortest action segment in the data given the data directory path 

all csvs should be located within the datatset dir path or in a subfolder
"""
def get_shortest_segment_length(data_dir_path):
    csv_paths = glob(data_dir_path + "/**/*.csv")
    csvs_df_list = []

    for csv_path in csv_paths:
        csvs_df_list.append(pd.read_csv(csv_path))

    all_csvs_df = pd.concat(csvs_df_list)

    end_times = all_csvs_df["End Time"].apply(timestamp_to_seconds)
    start_times = all_csvs_df["Start Time"].apply(timestamp_to_seconds)

    time_diffs = (end_times - start_times)
    
    # remove 0 second length segments (errors from labeling)
    if time_diffs.min() == 0:
        time_diffs.drop(time_diffs.index[time_diffs == 0].to_list(), inplace=True)
    
    return time_diffs.argmin()



"""
Post-process predictions.txt to get submission-ready text file
"""
def process_data(train_data_path):
    dir = os.getcwd() + "/slowfast/post_process"

    # delete existing post_processed_data.txt if exists
    txt_path = dir + "/post_processed_data.txt"
    if os.path.exists(txt_path):
        os.remove(txt_path)

    df = pd.read_csv(dir + "/predictions.txt", delimiter=" ", names=["video_id", "pred", "max_prob", "start_time", "end_time"])

    min_action_length = get_shortest_segment_length(train_data_path) # we will not use this since the shortest action label is 1 second...

    # threshold for usable preds
    prob_threshold = round(df["max_prob"].mean(), 1)

    # perform post-processing for each video_id (each id should have three videos, one for each camera angle)
    for video_id in sorted(df["video_id"].unique()):
        # get all video dfs with the same id
        video_id_df = df[df["video_id"] == video_id].reset_index()

        # parse the index for the first row of data for each video
        video_start_idxs  = video_id_df.index[video_id_df["start_time"] == 0].to_list()
        
        assert (len(video_start_idxs) == 3), "Error: more than three videos found per video id"

        # print(video_id_df.pivot_table(index = ["end_time"], aggfunc = "size"))

        for i, video_start_idx in enumerate(video_start_idxs):
            if(i != len(video_start_idxs) - 1):
                video_df = video_id_df.iloc[video_start_idx : video_start_idxs[i+1] - 1]
            else:
                video_df = video_id_df.iloc[video_start_idx : len(video_id_df) - 1]

            # filter out probs that don't meet threshold
            video_df.drop(video_df.index[video_df["max_prob"] < prob_threshold])

            # aggregate segments with same prob
            


if __name__ == '__main__':  

    A1_data_path = "/home/vislab-001/Jared/SET-A1"

    process_data(A1_data_path)
    


