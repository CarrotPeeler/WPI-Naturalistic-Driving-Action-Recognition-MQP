import os
import pandas as pd
import numpy as np
import math
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
Aggregates prediction and localization data across three diff. camera angles/videos for ONE video_id
Returns aggregated dataframe with 5 cols (video_id, pred, max_prob, start_time, end_time)

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
        # print(f"\n\n{video_df['pred'].to_list()} =====> LEN: {len(video_df)}\n\n{list(map(lambda x:int(x*100), video_df['max_prob'].to_list()))}\n\n")
    
    # check each df has same length
    eq_len_check = 0
    for video_d in video_dfs:
        if len(video_d) == len(video_dfs[0]): 
            eq_len_check += 1

    # assert eq_len_check == len(video_dfs), f"Dataframe size mismatch among videos for video_id {video_id_df['video_id'][0]}: {video_dfs}"
    if eq_len_check != len(video_dfs): print(f"Dataframe size mismatch among videos for video_id {video_id_df['video_id'][0]}: {video_dfs}")

    # aggregate pred and max_prob results for each row (each proposal) across all 3 camera angles
    agg_preds = []
    agg_probs = []
    # cnt = 0
    for row_idx in range(len(video_dfs[0])):
        preds = []
        probs = []

        # create lists of probs and preds for current row among all videos
        for vid_df in video_dfs:
            preds.append(vid_df.iloc[[row_idx]]["pred"].to_list()[0])
            probs.append(vid_df.iloc[[row_idx]]["max_prob"].to_list()[0])

        preds = np.array(preds)
        probs = np.array(probs)

        # check if there is a common pred among candidates
        if(stats.mode(preds, keepdims=False)[1] > 1):
            # cnt += 1
            # validate the probs of each pred are high enough to not be coincidence
            mode_pred = stats.mode(preds, keepdims=False)[0]

            mode_pred_idxs = np.where(preds == mode_pred)[0]
            minority_pred_idxs = np.where(preds != mode_pred)[0]

            mode_probs = np.array([probs[z] for z in mode_pred_idxs])
            minority_probs = np.array([probs[j] for j in minority_pred_idxs])

            # # check if minority pred exists and its prob is higher than mode probs
            # if(len(minority_pred_idxs) > 0 and minority_probs.max() > mode_probs.max() and mode_probs.max() < prob_threshold):
            #     # print(f"{preds} ==> {probs}")
            #     minority_pred = np.array([preds[m] for m in minority_pred_idxs]).max()
            #     agg_preds.append(minority_pred)
            #     agg_probs.append(minority_probs.max())
            
            # # select common pred if mean of common pred probs >= threshold
            # else:
            agg_preds.append(mode_pred)
            agg_probs.append(mode_probs.mean())

        # no common pred => select highest prob pred among all three camera angles
        else:
            best_prob_idx = probs.argmax()
            agg_preds.append(preds[best_prob_idx])
            agg_probs.append(probs.max())

    # print(f"NUM OF COMMON PRED ROWS: {cnt}")

    # since video_id, start, and end time columns should be same length, copy those cols from first video_df
    video_ids = [video_id_df['video_id'][0]] * len(video_dfs[0])
    start_times = video_dfs[0]["start_time"].to_list()
    end_times = video_dfs[0]["end_time"].to_list()

    agg_df = pd.DataFrame()
    agg_df["video_id"] = video_ids
    agg_df["pred"] = agg_preds
    agg_df["max_prob"] = agg_probs
    agg_df["start_time"] = start_times
    agg_df["end_time"] = end_times

    return agg_df

    # print(f"\n\nFINAL: {agg_preds} =====> LEN: {len(agg_preds)}\n\n")

        


"""
Post-process predictions.txt to get submission-ready text file

raw_output_filepath: path of text file containing inference output
prob_threshold: float b/w 0 and 1 that indicates the minimum acceptable probability used for filtering; by default uses avg prediction prob
"""
def process_data(raw_output_filepath:str, train_data_path:str, prob_threshold:float=None):
    dir = os.getcwd() + "/post_process/submission_files"

    # delete existing post_processed_data.txt if exists
    submission_filepath = dir + "/post_processed_data.txt"
    if os.path.exists(submission_filepath):
        os.remove(submission_filepath)

    df = pd.read_csv(raw_output_filepath, delimiter=" ", names=["video_id", "pred", "max_prob", "start_time", "end_time"])

    min_action_length = get_shortest_segment_length(train_data_path) # we will not use this since the shortest action label is 1 second...

    # threshold for usable preds
    if prob_threshold is None: 
        prob_threshold = round(df["max_prob"].mean(), 2)
        print(f"USING MEAN PRED PROB OF {prob_threshold} AS FILTERING THRESHOLD")
    # perform post-processing for each video_id (each id should have three videos, one for each camera angle)
    for idx, video_id in tqdm(enumerate(sorted(df["video_id"].unique())), total=len(df["video_id"].unique())):
        # get all video dfs with the same id
        video_id_df = df[df["video_id"] == video_id].reset_index()

        agg_df = aggregate_video_data(video_id_df, prob_threshold)

        if(idx == 0):
            agg_df.to_csv(submission_filepath)

        # print(video_id_df.pivot_table(index = ["end_time"], aggfunc = "size"))

        # filter out probs that don't meet threshold
        agg_df.drop(agg_df.index[agg_df["max_prob"] < prob_threshold], inplace=True)
        agg_df.reset_index(inplace=True)

        # aggregate segments with same prob
        merged_idxs = get_merged_segment_idxs(agg_df)

        for merged_idx_tuple in merged_idxs:
            merge_start_row = agg_df.loc[[merged_idx_tuple[0]]]
            merge_end_row = agg_df.loc[[merged_idx_tuple[1]]]

            activity_id = merge_start_row["pred"].to_list()[0]
            start_time = math.floor(merge_start_row["start_time"].to_list()[0])
            end_time = math.ceil(merge_end_row["end_time"].to_list()[0])

            with open(submission_filepath, "a+") as f:
                f.writelines(f"{video_id} {activity_id} {start_time} {end_time}\n")
            
            
if __name__ == '__main__':  

    A1_data_path = "/home/vislab-001/Jared/SET-A1"
    raw_output_filepath = '/home/vislab-001/Jared/Naturalistic-Driving-Action-Recognition-MQP/slowfast/post_process/mvitv2-b32x3/unprompted/predictions_unprompted_no_overlap.txt'

    process_data(raw_output_filepath, A1_data_path, prob_threshold=None)
    


