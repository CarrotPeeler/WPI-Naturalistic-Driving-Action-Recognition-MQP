import torch
import numpy as np
import pandas as pd
from scipy import stats


def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long().to("cuda")
    frames = torch.index_select(frames, 0, index)
    return frames


def predict_cam_views(cfg, model, model_2, cam_view_clips, agg_threshold, logger, resample=False):
    all_cam_view_preds = []
    all_cam_view_probs = []
    all_cam_view_preds_2 = []
    all_cam_view_probs_2 = []

    for cam_view_type in cam_view_clips.keys():
        if(cam_view_type == "Dashboard"): logger.info(f"NUM FRAMES AGGREGATED: {cam_view_clips[cam_view_type].shape[0]}")

        # subsample 16 frames from clip agg for each cam angle, then make pred
        if cam_view_clips[cam_view_type].shape[0] == cfg.DATA.NUM_FRAMES:
            input = [cam_view_clips[cam_view_type].permute(1,0,2,3).unsqueeze(dim=0)]

        elif cam_view_clips[cam_view_type].shape[0] > cfg.DATA.NUM_FRAMES and resample == False:
            # start_idx = 0
            # end_idx = start_idx + cam_view_clips[cam_view_type].shape[0] - 1
            # sampled = temporal_sampling(cam_view_clips[cam_view_type], start_idx, end_idx, cfg.DATA.NUM_FRAMES)
            # input = [sampled.permute(1,0,2,3).unsqueeze(dim=0)]

            # evenly sample half the num of input frames from among last half of frames in clip aggregation pool
            start_idx_1 = cam_view_clips[cam_view_type].shape[0] - agg_threshold
            end_idx_1 = start_idx_1 + cam_view_clips[cam_view_type].shape[0] - 1 - cfg.DATA.NUM_FRAMES
            sampled_1 = temporal_sampling(cam_view_clips[cam_view_type], start_idx_1, end_idx_1, int(cfg.DATA.NUM_FRAMES*cfg.TAL.AGG_SAMPLING_RATIO))

            # evenly sample half the num of input frames from last clip (most recently added proposal) in aggregation pool
            start_idx_2 = cam_view_clips[cam_view_type].shape[0] - cfg.DATA.NUM_FRAMES
            end_idx_2 = start_idx_2 + cam_view_clips[cam_view_type].shape[0] - 1
            sampled_2 = temporal_sampling(cam_view_clips[cam_view_type], start_idx_2, end_idx_2, int(cfg.DATA.NUM_FRAMES*cfg.TAL.SINGLE_PROP_SAMPLING_RATIO))

            sampled = torch.cat([sampled_1, sampled_2], dim=0)

            # aggregated input
            input = [sampled.permute(1,0,2,3).unsqueeze(dim=0)]

            # single clip input
            dev = 'cuda:1' if cfg.TAL.USE_2_GPUS == True else 'cuda:0'
            input_2 = [cam_view_clips[cam_view_type][start_idx_2:].permute(1,0,2,3).unsqueeze(dim=0).to(dev)] 

            cam_view_preds_2 = model_2(input_2).cpu()
            cam_view_pred_2 = cam_view_preds_2.argmax().item()
            cam_view_prob_2 = cam_view_preds_2.max().item()
            logger.info(f"PROP: {cam_view_type}, pred: {cam_view_pred_2}, prob: {cam_view_prob_2:.3f}")

            all_cam_view_preds_2.append(cam_view_pred_2)
            all_cam_view_probs_2.append(cam_view_prob_2)

        elif cam_view_clips[cam_view_type].shape[0] > cfg.DATA.NUM_FRAMES and resample == True:
            # assumes uniform sampling of new proposal failed -> resample new proposal but only select frames from 2nd half of clip
            start_idx_1 = cam_view_clips[cam_view_type].shape[0] - agg_threshold
            end_idx_1 = start_idx_1 + cam_view_clips[cam_view_type].shape[0] - 1 - cfg.DATA.NUM_FRAMES
            sampled_1 = temporal_sampling(cam_view_clips[cam_view_type], start_idx_1, end_idx_1, int(cfg.DATA.NUM_FRAMES*cfg.TAL.AGG_SAMPLING_RATIO))

            start_idx_2 = cam_view_clips[cam_view_type].shape[0] - int(cfg.DATA.NUM_FRAMES/2)
            end_idx_2 = start_idx_2 + cam_view_clips[cam_view_type].shape[0] - 1
            sampled_2 = cam_view_clips[cam_view_type][start_idx_2:end_idx_2]

            sampled = torch.cat([sampled_1, sampled_2], dim=0)

            # aggregated input
            input = [sampled.permute(1,0,2,3).unsqueeze(dim=0)]

        cam_view_preds = model(input).cpu()
        cam_view_pred = cam_view_preds.argmax().item()
        cam_view_prob = cam_view_preds.max().item()
        logger.info(f"SING: {cam_view_type}, pred: {cam_view_pred}, prob: {cam_view_prob:.3f}")

        all_cam_view_preds.append(cam_view_pred)
        all_cam_view_probs.append(cam_view_prob)

    preds = np.array(all_cam_view_preds)
    probs = np.array(all_cam_view_probs)
    preds_2 = np.array(all_cam_view_preds_2)
    probs_2 = np.array(all_cam_view_probs_2)

    return preds, probs, preds_2, probs_2


# consolidates predictions from all camera angles for a single proposal
def consolidate_preds(preds:np.array, probs:np.array, filtering_threshold:float, logger):
    """ 
    consol_code has 3 values: 
        0: common pred, non-tossed prediction
       -1: common pred, tossed prediction
       -2: no common pred, tossed prediction
    """

    # check if there is a common pred among candidates
    prediction_mode_stats = stats.mode(preds, keepdims=False)

    # count num of predictions matching mode and check if greater than 1 (meaning mode exists)
    if(prediction_mode_stats[1] > 1):
        # retrieve common pred
        mode_pred = prediction_mode_stats[0]

        # retrieve indexes of original array where predictions match the common pred
        mode_pred_idxs = np.where(preds == mode_pred)[0]

        # retrieve the corresponding probabilities for each prediction that matches the common pred
        mode_probs = np.array([probs[z] for z in mode_pred_idxs])

        # if all three predictions are identical, filter out lowest pred prob among them
        if prediction_mode_stats[1] == 3:
            mode_probs = np.delete(mode_probs, np.argmin(mode_probs))
            
        # calc mean prob of common preds
        agg_prob = mode_probs.mean()

        # common pred
        curr_agg_pred = mode_pred

        # even if there's a common pred among camera angles, check if mean of their probs is lower than threshold
        if agg_prob >= filtering_threshold:
            consol_code = 0
        else:
            consol_code = -1

        logger.info(f"agg pred: {curr_agg_pred}, agg prob: {agg_prob:.3f}")

    # no common pred => select highest prob pred among all three camera angles
    else:
        consol_code = -2

        best_prob_idx = probs.argmax()
        curr_agg_pred = preds[best_prob_idx]
        agg_prob = probs.max()

        logger.info(f"agg pred: {curr_agg_pred}, agg prob: {agg_prob:.3f}")

    return curr_agg_pred, consol_code


"""
returns list of tuples (start_row_idx, end_row_idx) for rows of the original df that should be merged into one row
(start_row_idx, end_row_idx): start and end indices of rows that are consecutive and have the same pred
"""
def get_merged_segment_idxs(video_df):
    merged_idxs = []

    row_idx = 0
    while row_idx < len(video_df):
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

        # assert list of idxs is consec
        assert sorted(consec_pred_idxs) == list(range(min(consec_pred_idxs), max(consec_pred_idxs)+1)), "Elements not consecutive"

        merged_idxs.append((consec_pred_idxs[0], consec_pred_idxs[-1]))

    return merged_idxs


"""
Merges consecutive temporal intervals with same prediction

params:
    path_to_txt: path to txt file storing unmerged submission results
"""
def post_process_merge(path_to_txt, submission_filepath):
    df = pd.read_csv(path_to_txt, sep=" ", names=['video_id', 'pred', 'start_time', 'end_time'])

    # get merged idxs
    merged_idxs = get_merged_segment_idxs(df)

    for merged_idx_tuple in merged_idxs:
        merge_start_row = df.loc[[merged_idx_tuple[0]]]
        merge_end_row = df.loc[[merged_idx_tuple[1]]]

        video_id = merge_start_row["video_id"].to_list()[0]
        activity_id = merge_start_row["pred"].to_list()[0]
        start_time = merge_start_row["start_time"].to_list()[0]
        end_time = merge_end_row["end_time"].to_list()[0]

        with open(submission_filepath, "a+") as f:
                f.writelines(f"{video_id} {activity_id} {start_time} {end_time}\n")