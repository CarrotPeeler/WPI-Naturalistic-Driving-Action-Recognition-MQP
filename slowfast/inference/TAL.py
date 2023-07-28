import torch
import numpy as np
import pandas as pd
from scipy import stats
from torchvision.utils import save_image


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


def predict_cam_views(cfg, model, model_2, cam_view_clips, agg_threshold, logger, resample=False, cur_iter=None):
    all_cam_view_probs = {}
    all_cam_view_probs_2 = {}

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
            start_idx_1 = 0 # cam_view_clips[cam_view_type].shape[0] - agg_threshold
            end_idx_1 = start_idx_1 + cam_view_clips[cam_view_type].shape[0] - 1 - cfg.DATA.NUM_FRAMES
            sampled_1 = temporal_sampling(cam_view_clips[cam_view_type], start_idx_1, end_idx_1, int(cfg.DATA.NUM_FRAMES*cfg.TAL.AGG_SAMPLING_RATIO))

            SINGLE_PROP_SAMPLING_RATIO = 1.0 - cfg.TAL.AGG_SAMPLING_RATIO

            # evenly sample half the num of input frames from last clip (most recently added proposal) in aggregation pool
            start_idx_2 = cam_view_clips[cam_view_type].shape[0] - cfg.DATA.NUM_FRAMES
            end_idx_2 = start_idx_2 + cam_view_clips[cam_view_type].shape[0] - 1
            sampled_2 = temporal_sampling(cam_view_clips[cam_view_type], start_idx_2, end_idx_2, int(cfg.DATA.NUM_FRAMES*SINGLE_PROP_SAMPLING_RATIO))

            sampled = torch.cat([sampled_1, sampled_2], dim=0)

            # aggregated input
            input = [sampled.permute(1,0,2,3).unsqueeze(dim=0)]

            # single clip input
            dev = 'cuda:1' if cfg.TAL.USE_2_GPUS == True else 'cuda:0'
            sampled_3 = cam_view_clips[cam_view_type][start_idx_2:]
            input_2 = [sampled_3.permute(1,0,2,3).unsqueeze(dim=0).to(dev)] 

            cam_view_preds_2 = model_2(input_2).cpu()
            cam_view_probs_2 = cam_view_preds_2.numpy()
            all_cam_view_probs_2[cam_view_type] = cam_view_probs_2

            # if cur_iter == 56: #and cur_iter <= 59:
            #     save_image(sampled, f"{cfg.PROMPT.IMAGE_FOLDER}/input_1_no_resample_iter_{cur_iter}.png")
            #     save_image(sampled_3, f"{cfg.PROMPT.IMAGE_FOLDER}/input_2_no_resample_iter_{cur_iter}.png")

        elif cam_view_clips[cam_view_type].shape[0] > cfg.DATA.NUM_FRAMES and resample == True:
            # assumes uniform sampling of new proposal failed -> resample new proposal but only select frames from 2nd half of clip
            start_idx_1 = 0 # cam_view_clips[cam_view_type].shape[0] - agg_threshold
            end_idx_1 = start_idx_1 + cam_view_clips[cam_view_type].shape[0] - 1 - cfg.DATA.NUM_FRAMES
            sampled_1 = temporal_sampling(cam_view_clips[cam_view_type], start_idx_1, end_idx_1, int(cfg.DATA.NUM_FRAMES*cfg.TAL.AGG_SAMPLING_RATIO))

            start_idx_2 = cam_view_clips[cam_view_type].shape[0] - int(cfg.DATA.NUM_FRAMES/2)
            end_idx_2 = start_idx_2 + cam_view_clips[cam_view_type].shape[0] - 1
            sampled_2 = cam_view_clips[cam_view_type][start_idx_2:end_idx_2]

            sampled = torch.cat([sampled_1, sampled_2], dim=0)

            # aggregated input
            input = [sampled.permute(1,0,2,3).unsqueeze(dim=0)]

            # if cur_iter == 56: #and cur_iter <= 59:
            #     save_image(sampled, f"{cfg.PROMPT.IMAGE_FOLDER}/input_1_resampled_iter_{cur_iter}.png")

        cam_view_preds = model(input).cpu()
        cam_view_probs = cam_view_preds.numpy()
        all_cam_view_probs[cam_view_type] = cam_view_probs

    return all_cam_view_probs, all_cam_view_probs_2


# consolidates predictions from all camera angles for a single proposal
def consolidate_preds(cam_view_probs:dict, cam_view_weights:dict, filtering_threshold:float, logger):
    consolidated_probs = cam_view_weights['Dashboard'] * cam_view_probs['Dashboard']\
                       + cam_view_weights['Rear_view'] * cam_view_probs['Rear_view']\
                       + cam_view_weights['Right_side_window'] * cam_view_probs['Right_side_window']
    
    consolidated_pred = np.argmax(consolidated_probs)
    consolidated_prob = np.max(consolidated_probs)

    consol_code = -1 if consolidated_prob < filtering_threshold else 0

    logger.info(f"AGG pred: {consolidated_pred}, prob: {consolidated_prob:.3f}, code: {consol_code}")

    return consolidated_pred, consol_code


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

                # fetch future start and current end timestamps
                if i < len(same_pred_idxs) - 1:
                    next_interval_start = video_df.iloc[[same_pred_idxs[i+1]]]["start_time"].to_list()[0]
                    curr_interval_end = video_df.iloc[[same_pred_row_idx]]["end_time"].to_list()[0]

                if(i == len(same_pred_idxs) - 1 or same_pred_idxs[i+1] - same_pred_row_idx != 1 or next_interval_start < curr_interval_end):
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