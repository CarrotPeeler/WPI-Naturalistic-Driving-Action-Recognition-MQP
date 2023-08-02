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


"""
Performs prediction step over all three camera views for a given set or aggregation of frames

params:
    cfg: cfg object with hyperparams
    model and model_2: both are pytorch models, which use the same checkpoint but run on diff. GPUs
    cam_view_clips: dict containing batches of frames for each camera view
    logger: logger object for printing
    resample: whether to resample current temporal interval 1s later than when it starts

returns:
    two dicts (1 for aggregated frames, 1 for current temporal interval's frames), each containing 3 prob matrices for each camera view
"""
def predict_cam_views(cfg, model, model_2, cam_view_clips, agg_threshold, logger, resample=False, cur_iter=None):
    all_cam_view_probs = {}
    all_cam_view_probs_2 = {}

    for cam_view_type in cam_view_clips.keys():
        if(cam_view_type == "Dashboard") and cfg.TAL.PRINT_DEBUG_OUTPUT: logger.info(f"NUM FRAMES AGGREGATED: {cam_view_clips[cam_view_type].shape[0]}")

        # subsample 16 frames from clip agg for each cam angle, then make pred
        if cam_view_clips[cam_view_type].shape[0] == cfg.DATA.NUM_FRAMES:
            input = [cam_view_clips[cam_view_type].permute(1,0,2,3).unsqueeze(dim=0)]

        elif cam_view_clips[cam_view_type].shape[0] > cfg.DATA.NUM_FRAMES and resample == False:
            # evenly sample half the num of input frames from among last half of frames in clip aggregation pool
            start_idx_1 = 0 
            end_idx_1 = cam_view_clips[cam_view_type].shape[0] - 1 - cfg.DATA.NUM_FRAMES
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
            start_idx_1 = 0 
            end_idx_1 = cam_view_clips[cam_view_type].shape[0] - 1 - cfg.DATA.NUM_FRAMES
            sampled_1 = temporal_sampling(cam_view_clips[cam_view_type], start_idx_1, end_idx_1, int(cfg.DATA.NUM_FRAMES*cfg.TAL.AGG_SAMPLING_RATIO))

            start_idx_2 = cam_view_clips[cam_view_type].shape[0] - int(cfg.DATA.NUM_FRAMES/2)
            sampled_2 = cam_view_clips[cam_view_type][start_idx_2:]

            sampled = torch.cat([sampled_1, sampled_2], dim=0)

            # aggregated input
            input = [sampled.permute(1,0,2,3).unsqueeze(dim=0)]

            # if cur_iter == 56: #and cur_iter <= 59:
            #     save_image(sampled, f"{cfg.PROMPT.IMAGE_FOLDER}/input_1_resampled_iter_{cur_iter}.png")

        cam_view_preds = model(input).cpu()
        cam_view_probs = cam_view_preds.numpy()
        all_cam_view_probs[cam_view_type] = cam_view_probs

    return all_cam_view_probs, all_cam_view_probs_2


"""
Re-performs predictions over short segment ~6s or less, as a means of strengthening classification confidence
Specifically samples frames from overlapping intervals that have not been sampled and predicted on already

params:
    cfg: cfg object with hyperparams
    model: model to use for this operation
    cam_view_clips: dict containing batches of frames for each camera view

returns:
    list of dicts (1 for each sampled interval), each containing 3 probability matrices, 1 for each camera view
"""
def predict_short_segment(cfg, model, cam_view_clips):
    dev = 'cuda:1' if cfg.TAL.USE_2_GPUS == True else 'cuda:0'

    all_segment_probs = []
    segment_sample_idxs = []
    # do not include the most recently added frames, they may contain a diff action and low probs
    num_total_frames = cam_view_clips['Dashboard'].shape[0] - cfg.DATA.NUM_FRAMES
    sample_stride = cfg.DATA.NUM_FRAMES//4

    for start_idx in range(0, num_total_frames, sample_stride):
        all_cam_view_probs = {}
        end_idx = start_idx + cfg.DATA.NUM_FRAMES

        # if start_idx % cfg.DATA.NUM_FRAMES != 0:
        for cam_view_type in cam_view_clips.keys():
            sampled = cam_view_clips[cam_view_type][start_idx:end_idx]
            input = [sampled.permute(1,0,2,3).unsqueeze(dim=0).to(dev)]

            cam_view_preds = model(input).cpu()
            cam_view_probs = cam_view_preds.numpy()
            all_cam_view_probs[cam_view_type] = cam_view_probs

        all_segment_probs.append(all_cam_view_probs)
        segment_sample_idxs.append(start_idx)

    return all_segment_probs, segment_sample_idxs


"""
Given proposal prob mats for a single localized action interval and prob mats for the short segment re-evaluation,
as well as their respective starting frame index, re-order the prob mats according to their temporal boundaries

params:
    non_overlap_prob_mats: prob matrices accumulated from non-overlap sampling
    overlap_prob_mats: prob matrices obtained from short-seg re-eval (overlap sampling)
    overlap_sample_idxs: idxs corresponding to the first frame of each sampled overlapping interval

returns:
    list of prob mats, reordered temporal idx
"""
def get_reordered_prob_mats(cfg, non_overlap_prob_mats, overlap_prob_mats, overlap_sample_idxs):
    # generate start idxs for non-overlapping prob mats
    non_overlap_sample_idxs = [i*cfg.DATA.NUM_FRAMES for i in range(len(non_overlap_prob_mats))]
    sample_idxs = non_overlap_sample_idxs + overlap_sample_idxs

    prob_mats = non_overlap_prob_mats + overlap_prob_mats

    _, reordered_prob_mats = (list(t) for t in zip(*sorted(zip(sample_idxs, prob_mats))))

    return reordered_prob_mats


"""
Generates Gaussian weights

params:
    sigma: sigma term of Gaussian filter
    length: window size (number of samples to consider for filtering)

returns:
    NParray of Gaussian weights for a window size of 'length'
"""
def generate_gaussian_weights(sigma, length):
    center = length // 2
    x = np.linspace(-center, center, length)
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    return kernel


"""
Removes all action intervals that do not repeat consecutively and occur in between two actions of the same type
Classes which the model is very sensitive to (4, 11, 12) have special filtering conditions to improve classification confidence
    > conditions based on common mistakes model makes
"""
def filter_noisy_actions(prev_pred, prob_mats, segment_preds):
    for i, prob_mat in enumerate(prob_mats):
        if prev_pred == 4 and np.array(prob_mat).argmax() == 0:
            del prob_mats[i]
            del segment_preds[i]

        elif prev_pred == 11 and np.array(prob_mat).argmax() == 12:
            del prob_mats[i]

        elif prev_pred == 12 and np.array(prob_mat).argmax() in [4,11]:
            del prob_mats[i]

        elif i > 0 and i + 1 < len(prob_mats):
            past = np.array(prob_mats[i-1]).argmax()
            present = np.array(prob_mat).argmax()
            future = np.array(prob_mats[i+1]).argmax()

            if past != present and present != future and past == future and present != prev_pred:
                del prob_mats[i]


""" 
Consolidates multiple action prob matrices by computing the Gaussian weighted average

params:
    consolidated_prob_mats: a list of prob mats (ordered temporally) for each sampled interval
    sigma: sigma term in Gaussian filtering equation
    filtering_threshold: list of thresholds for each action id for filtering bad probs 

returns:
    final prediction and validity code determined by Gaussian weighted average
"""
def consolidate_cum_preds_with_gaussian(cfg, consolidated_prob_mats:list, prev_agg_pred, segment_preds, sigma, filtering_thresholds, logger):
    filter_noisy_actions(prev_agg_pred, consolidated_prob_mats, segment_preds)
    prob_mats = np.vstack(consolidated_prob_mats)

    weights = generate_gaussian_weights(sigma, len(prob_mats))
    weighted_prob_mats = []

    for i, prob_mat in enumerate(prob_mats):
        weighted_prob_mats.append(prob_mat * weights[i])

    gaussian_avged_mat = np.sum(weighted_prob_mats, axis=0) / np.sum(weights, axis=0)

    final_prob = np.max(gaussian_avged_mat)
    final_pred = np.argmax(gaussian_avged_mat)

    code = 0
    # check for false positives below threshold or special cases (hints of false positives among high passing probs)
    if final_prob < filtering_thresholds[final_pred]\
        or (final_pred in [3,6] and any(pred not in [0,3,6] for pred in set(segment_preds)))\
        or (final_pred in [2,5] and any(pred not in [0,2,5] for pred in set(segment_preds)))\
        or final_pred == 4 and len(set(segment_preds)) > 2:
        code = -1

    if cfg.TAL.PRINT_DEBUG_OUTPUT: logger.info(f"mats: {prob_mats}, Gaussian mat: {gaussian_avged_mat}, final prob: {final_prob:.3f}")

    return final_pred, code

"""
Same as Gaussian method but applies equal weight to all sampled interval probs
"""
def consolidate_cum_preds_with_mean(cfg, consolidated_prob_mats: list, filtering_thresholds, logger):
    consolidated_prob_mats = np.vstack(consolidated_prob_mats)
    avged_mat = consolidated_prob_mats.mean(axis=0)

    final_prob = np.max(avged_mat)
    final_pred = np.argmax(avged_mat)

    if final_prob < filtering_thresholds[final_pred]:
        code = -1
    else:
        code = 0

    if cfg.TAL.PRINT_DEBUG_OUTPUT: logger.info(f"mats: {consolidated_prob_mats}, mat: {avged_mat}, final prob: {final_prob:.3f}")

    return final_pred, code

       
"""
Consolidates predictions from all camera angles for a single proposal

returns:
    the final prediction and the code indicating whether it is valid (passes threshold) or invalid
"""
def consolidate_preds(cfg, cam_view_probs:dict, cam_view_weights:dict, filtering_threshold:float, logger):
    consolidated_probs = cam_view_weights['Dashboard'] * cam_view_probs['Dashboard']\
                       + cam_view_weights['Rear_view'] * cam_view_probs['Rear_view']\
                       + cam_view_weights['Right_side_window'] * cam_view_probs['Right_side_window']
    
    consolidated_pred = np.argmax(consolidated_probs)
    consolidated_prob = np.max(consolidated_probs)

    consol_code = -1 if consolidated_prob < filtering_threshold else 0

    if cfg.TAL.PRINT_DEBUG_OUTPUT: logger.info(f"AGG pred: {consolidated_pred}, prob: {consolidated_prob:.3f}, code: {consol_code}")

    return consolidated_pred, consol_code, consolidated_probs


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