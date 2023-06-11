import os
import pandas as pd
import numpy as np
from glob import glob
from scipy import stats
from tqdm.auto import tqdm
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
returns list of tuples (start_row_idx, end_row_idx) for rows of the original df that should be merged into one row
(start_row_idx, end_row_idx): start and end indices of rows that are consecutive and have the same pred
"""
def get_merged_segment_idxs(video_df):
    merged_idxs = []

    row_idx = 0
    # count = 0
    while row_idx < len(video_df):
        # count += 1
        # get row pred and then find all other row idxs with same pred
        pred_class = video_df.iloc[[row_idx]]["pred"].to_list()[0]
        same_pred_idxs = video_df.index[video_df["pred"] == pred_class].to_list()

        # find consecutive row idxs with same pred
        consec_pred_idxs = []
        for i, same_pred_row_idx in enumerate(same_pred_idxs):
            if(same_pred_row_idx >= row_idx):
                consec_pred_idxs.append(same_pred_row_idx)
            
                # print(f"idx: {same_pred_row_idx} => {same_pred_idxs[i+1] - same_pred_row_idx} not end?: {same_pred_row_idx != len(same_pred_idxs) - 1}")
                if(i == len(same_pred_idxs) - 1 or same_pred_idxs[i+1] - same_pred_row_idx != 1):
                    row_idx = same_pred_row_idx+1
                    break
            
        # print(consec_pred_idxs)

        # assert list of idxs is consec
        assert sorted(consec_pred_idxs) == list(range(min(consec_pred_idxs), max(consec_pred_idxs)+1)), "Elements not consecutive"
      
        # if(count == 3): row_idx = len(video_df)
        merged_idxs.append((consec_pred_idxs[0], consec_pred_idxs[-1]))
    
    return merged_idxs



"""
Aggregates prediction and localization data across three diff. camera angles/videos
Returns aggregated dataframe with 4 cols (video_id, activity_id, start_time, end_time)

video_id_df: dataframe containing raw output inference data for ONE video_id (may contain multiple videos)
"""
def aggregate_video_data(video_id_df: pd.DataFrame, prob_threshold:float):
    # parse the index for the first row of data for each video
    video_start_idxs  = video_id_df.index[video_id_df["start_time"] == 0].to_list()
    
    assert (len(video_start_idxs) == 3), "Error: more than three videos found per video id"

    # store dfs for each vid in array
    video_dfs = []
    for i, video_start_idx in enumerate(video_start_idxs):
        if(i != len(video_start_idxs) - 1):
            video_df = video_id_df.iloc[video_start_idx : video_start_idxs[i+1] - 1]
        else:
            video_df = video_id_df.iloc[video_start_idx : len(video_id_df) - 1]

        video_dfs.append(video_df)
        print(f"\n\n{video_df['pred'].to_list()} =====> LEN: {len(video_df)}\n\n{list(map(lambda x:int(x*10), video_df['max_prob'].to_list()))}\n\n")
    
    # check each df has same length
    eq_len_check = 0
    for video_d in video_dfs:
        if len(video_d) == len(video_dfs[0]): 
            eq_len_check += 1

    assert eq_len_check == len(video_dfs), "Dataframe size mismatch for video_id"

    # aggregate results by selecting highest prob pred for each pred among all videos
    agg_preds = []
    for row_idx in range(len(video_dfs[0])):
        preds = []
        probs = []

        # create lists of probs and preds for current row among all videos
        for vid_df in video_dfs:
            preds.append(vid_df.iloc[[row_idx]]["pred"].to_list()[0])
            probs.append(vid_df.iloc[[row_idx]]["max_prob"].to_list()[0])

        preds = np.array(preds)
        probs = np.array(probs)
            
        # first check if there is a common pred among candidates
        if(stats.mode(preds, keepdims=False)[1] > 1):
            # validate the probs of each pred are high enough to not be coincidence
            mode_pred = stats.mode(preds, keepdims=False)[0]
            mode_pred_idxs = np.where(preds == mode_pred)
            mode_probs = np.array([probs[z] for z in mode_pred_idxs])
            
            if(mode_probs.mean() >= prob_threshold):
                agg_preds.append(mode_pred)

        # no common pred => find highest prob pred and append
        else:
            agg_preds.append(preds[probs.argmax()])

    print(f"\n\nFINAL: {agg_preds} =====> LEN: {len(agg_preds)}\n\n")

        


"""
Post-process predictions.txt to get submission-ready text file

raw_output_filepath: path of text file containing inference output
"""
def process_data(raw_output_filepath:str, train_data_path:str):
    dir = os.getcwd() + "/post_process"

    # delete existing post_processed_data.txt if exists
    txt_path = dir + "/post_processed_data.txt"
    if os.path.exists(txt_path):
        os.remove(txt_path)

    df = pd.read_csv(raw_output_filepath, delimiter=" ", names=["video_id", "pred", "max_prob", "start_time", "end_time"])

    min_action_length = get_shortest_segment_length(train_data_path) # we will not use this since the shortest action label is 1 second...

    # threshold for usable preds
    prob_threshold = round(df["max_prob"].mean(), 1)

    # column values for final dataframe
    video_id_col = []
    activity_id_col = []
    start_time_col = []
    end_time_col = []

    # perform post-processing for each video_id (each id should have three videos, one for each camera angle)
    for idx, video_id in tqdm(enumerate(sorted(df["video_id"].unique()))):
        # get all video dfs with the same id
        video_id_df = df[df["video_id"] == video_id].reset_index()

        if video_id == 1: aggregate_video_data(video_id_df, prob_threshold)

        # print(video_id_df.pivot_table(index = ["end_time"], aggfunc = "size"))

       


        # # filter out probs that don't meet threshold
        # video_df.drop(video_df.index[video_df["max_prob"] < prob_threshold], inplace=True)
        # video_df.reset_index(inplace=True)

        # # aggregate segments with same prob
        # merged_idxs = get_merged_segment_idxs(video_df)

        # # column values for this video's merged preds
        # temp_video_id_col = [video_id] * len(merged_idxs)
        # temp_activity_id_col = []
        # temp_start_time_col = []
        # temp_end_time_col = []

        # for merged_idx_tuple in merged_idxs:
        #     merge_start_row = video_df.loc[[merged_idx_tuple[0]]]
        #     merge_end_row = video_df.loc[[merged_idx_tuple[1]]]

        #     temp_activity_id_col.append(merge_start_row["pred"].to_list()[0])
        #     temp_start_time_col.append(merge_start_row["start_time"].to_list()[0])
        #     temp_end_time_col.append(merge_end_row["end_time"].to_list()[0])

        # video_id_data_dict["video_id_cols"].append(temp_video_id_col)
        # video_id_data_dict["activity_id_cols"].append(temp_activity_id_col)
        # video_id_data_dict["start_time_cols"].append(temp_start_time_col)
        # video_id_data_dict["end_time_cols"].append(temp_end_time_col)

        # append aggregated video_id data to col lists
        # video_id_col.extend()
        # activity_id_col.extend()
        # start_time_col.extend()
        # end_time_col.extend()
            
            

if __name__ == '__main__':  

    A1_data_path = "/home/vislab-001/Jared/SET-A1"
    raw_output_filepath = "/home/vislab-001/Jared/Naturalistic-Driving-Action-Recognition-MQP/slowfast/post_process/mvitv2-b32x3/normal_data/predictions_mvitv2-b_240_epochs_no_overlap.txt"

    process_data(raw_output_filepath, A1_data_path)
    


