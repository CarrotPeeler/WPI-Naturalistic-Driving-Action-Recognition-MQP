import os
import torch
import decord
import random
import pandas as pd
import numpy as np
import slowfast.utils.checkpoint as cu
from slowfast.models import build_model
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import List
from slowfast.datasets import utils as utils
from slowfast.datasets.random_erasing import RandomErasing
from slowfast.datasets.transform import (
    MaskingGenerator,
    MaskingGenerator3D,
    create_random_augment,
)


class VideoProposalDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, video_path, frame_length, frame_stride, proposal_stride, num_workers=1):
        self.cfg = cfg
        self.aug = False
        self.rand_erase = False

        self.video_path = video_path
        self.frame_length = frame_length
        self.transform = transform
        self.proposals = []

        # decode video and start proposal generation        
        frames = decord.VideoReader(self.video_path, num_threads=num_workers)
        proposal_length = frame_length * frame_stride

        # list of proposal tuples (start_frame_idx, end_frame_idx)
        temp_proposals = self.generate_proposals(proposal_stride, proposal_length, len(frames))

        # modify proposal tuples to add sampled frames (sampled_frames, start_frame_idx, end_frame_idx)
        for i in tqdm(range(len(temp_proposals))):
            temp_proposal = temp_proposals[i]
            self.proposals.append((self.temporal_sampling(frames, temp_proposal[0], temp_proposal[1], self.frame_length),
                                   temp_proposal[0],
                                   temp_proposal[1]))
        
    """
    Returns list of proposal tuples without frames (start_frame_idx, end_frame_idx)
    """
    def generate_proposals(self, proposal_stride, proposal_length, video_frame_count):
        proposals = []
        for i in range(0, video_frame_count - proposal_stride - 1, proposal_stride):
            # each proposal is a tuple with a start and end index
            proposals.append((i, i+proposal_length-1))
        return proposals
    
    """
    Returns temporally sampled frames given list of frames, start_frame_idx, end_frame_idx, number of frames to sample
    """
    def temporal_sampling(self, frames, start_frame_idx, end_frame_idx, num_samples):
        frames_batch = frames.get_batch(list(range(start_frame_idx, end_frame_idx + 1))).asnumpy()

        idxs = torch.linspace(0, len(frames_batch) - 1, num_samples)
        idxs = torch.clamp(idxs, 0, len(frames_batch) - 1).long()
        
        sampled_frames = torch.index_select(torch.from_numpy(frames_batch), 0, idxs)
        return sampled_frames
    
    """
    Masking generation for frames
    """
    def _gen_mask(self):
        if self.cfg.AUG.MASK_TUBE:
            num_masking_patches = round(
                np.prod(self.cfg.AUG.MASK_WINDOW_SIZE) * self.cfg.AUG.MASK_RATIO
            )
            min_mask = num_masking_patches // 5
            masked_position_generator = MaskingGenerator(
                mask_window_size=self.cfg.AUG.MASK_WINDOW_SIZE,
                num_masking_patches=num_masking_patches,
                max_num_patches=None,
                min_num_patches=min_mask,
            )
            mask = masked_position_generator()
            mask = np.tile(mask, (8, 1, 1))
        elif self.cfg.AUG.MASK_FRAMES:
            mask = np.zeros(shape=self.cfg.AUG.MASK_WINDOW_SIZE, dtype=np.int)
            n_mask = round(
                self.cfg.AUG.MASK_WINDOW_SIZE[0] * self.cfg.AUG.MASK_RATIO
            )
            mask_t_ind = random.sample(
                range(0, self.cfg.AUG.MASK_WINDOW_SIZE[0]), n_mask
            )
            mask[mask_t_ind, :, :] += 1
        else:
            num_masking_patches = round(
                np.prod(self.cfg.AUG.MASK_WINDOW_SIZE) * self.cfg.AUG.MASK_RATIO
            )
            max_mask = np.prod(self.cfg.AUG.MASK_WINDOW_SIZE[1:])
            min_mask = max_mask // 5
            masked_position_generator = MaskingGenerator3D(
                mask_window_size=self.cfg.AUG.MASK_WINDOW_SIZE,
                num_masking_patches=num_masking_patches,
                max_num_patches=max_mask,
                min_num_patches=min_mask,
            )
            mask = masked_position_generator()
        return mask
        
    def __len__(self):
        "Returns the total number of proposals for this video."
        return len(self.proposals)

    # Code below mostly adapted from slowfast/slowfast/datasets/kinetics.py
    def __getitem__(self, index: int):
        "Returns one proposal (frames, start_frame_idx, end_frame_idx)."

        # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
        # center, or right if width is larger than height, and top, middle,
        # or bottom if height is larger than width.
        spatial_sample_index = 1

        min_scale, max_scale, crop_size = (
            [self.cfg.DATA.TEST_CROP_SIZE] * 3
            if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
            else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2
            + [self.cfg.DATA.TEST_CROP_SIZE]
        )
        min_scale, max_scale, crop_size = [min_scale], [max_scale], [crop_size]\
        
        # get proposal (frames, start_frame_idx, end_frame_idx)
        proposal = self.proposals[index]
        frames_decoded = proposal[0]

        num_decode = 1 # num of spatial crops
        num_aug = 1
        num_out = num_aug * num_decode
        
        f_out = [None] * num_out
        idx = -1

        for i in range(num_decode):
            for _ in range(num_aug):
                idx += 1
                f_out[idx] = frames_decoded[i].clone()
                f_out[idx] = f_out[idx].float()
                f_out[idx] = f_out[idx] / 255.0

                if self.aug and self.cfg.AUG.AA_TYPE:
                    aug_transform = create_random_augment(
                        input_size=(f_out[idx].size(1), f_out[idx].size(2)),
                        auto_augment=self.cfg.AUG.AA_TYPE,
                        interpolation=self.cfg.AUG.INTERPOLATION,
                    )
                    # T H W C -> T C H W.
                    f_out[idx] = f_out[idx].permute(0, 3, 1, 2)
                    list_img = self._frame_to_list_img(f_out[idx])
                    list_img = aug_transform(list_img)
                    f_out[idx] = self._list_img_to_frames(list_img)
                    f_out[idx] = f_out[idx].permute(0, 2, 3, 1)

                # Perform color normalization.
                f_out[idx] = utils.tensor_normalize(
                    f_out[idx], self.cfg.DATA.MEAN, self.cfg.DATA.STD
                )

                # T H W C -> C T H W.
                f_out[idx] = f_out[idx].permute(3, 0, 1, 2)

                scl, asp = (
                    self.cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE,
                    self.cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
                )
                relative_scales = None
                relative_aspect = None
                
                f_out[idx] = utils.spatial_sampling(
                    f_out[idx],
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale[i],
                    max_scale=max_scale[i],
                    crop_size=crop_size[i],
                    random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                    inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                    aspect_ratio=relative_aspect,
                    scale=relative_scales,
                    motion_shift=False,
                )

                if self.rand_erase:
                    erase_transform = RandomErasing(
                        self.cfg.AUG.RE_PROB,
                        mode=self.cfg.AUG.RE_MODE,
                        max_count=self.cfg.AUG.RE_COUNT,
                        num_splits=self.cfg.AUG.RE_COUNT,
                        device="cpu",
                    )
                    f_out[idx] = erase_transform(
                        f_out[idx].permute(1, 0, 2, 3)
                    ).permute(1, 0, 2, 3)

                f_out[idx] = utils.pack_pathway_output(self.cfg, f_out[idx])
                if self.cfg.AUG.GEN_MASK_LOADER:
                    mask = self._gen_mask()
                    f_out[idx] = f_out[idx] + [torch.Tensor(), mask]

        frames = f_out[0] if num_out == 1 else f_out
    
        # return proposal tuple with new frame modifications
        return (frames, proposal[1], proposal[2])
        


"""
Returns dict of video_ids and their corresponding video file names
"""
def get_video_ids_dict(path_to_csv):
    video_ids_dict = dict()
    df = pd.read_csv(path_to_csv)

    for idx, row in df.iterrows():
        key = int(row['video_id'])
        val = row[df.columns[1:4]].to_list()
        video_ids_dict[int(key)] = val

    return video_ids_dict



"""
Returns a loaded model with last saved checkpoint given a config file
"""
def load_model(cfg):
    model = build_model(cfg)
    cu.load_test_checkpoint(cfg, model)
    return model



"""
Returns a prediction generated from a model, given the model and a batch of frames as input
"""
def make_prediction(model, batch_frames):
    return model(batch_frames)



if __name__ == '__main__':

    ########### Configuration Params ############
    A2_data_path = "/home/vislab-001/Jared/SET-A2"
    frame_length = 16
    frame_stride = 4
    proposal_stride = frame_length * frame_stride # for non-overlapping proposals; set smaller num for overlapping 
    transform = None
    num_workers = 8 #os.cpu_count()
    batch_size = 1
    ############################################
    path_to_config = os.getcwd() + "/configs/SLOWFAST_8x8_R50_inf.yaml"

    video_ids_dict = get_video_ids_dict(os.getcwd() + "/inference/video_ids.csv")
    video_paths = glob(A2_data_path + "/**/*.MP4")
    args = parse_args()
    cfg = load_config(args, path_to_config)
    cfg = assert_and_infer_cfg(cfg)

    model = load_model(cfg)

    print(f"Number of threads: {num_workers}")

    for i in tqdm(range(len(video_paths))):
        proposals_dataset = VideoProposalDataset(cfg, video_paths[i], frame_length, frame_stride, proposal_stride, num_workers=num_workers)
        proposals_dataloader = DataLoader(dataset=proposals_dataset, batch_size=batch_size, num_workers=num_workers)

        video_name = video_paths[i].rpartition('/')[2]
        video_id =  {i for i in video_ids_dict if video_name in video_ids_dict[i]}
        video_id = list(video_id)[0]

        model.eval()
        with torch.inference_mode():
            for batch_idx, (batch_frames, start_frame_idxs, end_frame_idxs) in enumerate(proposals_dataloader):
                prediction = make_prediction(model, batch_frames)
                print(prediction)
                # # each batch has many proposals => we iterate through each proposal in a batch
                # for proposal_idx in enumerate(batch_frames.shape[0]):
                #     # write prob, start_frame_idx, end_frame_idx to file
                #     with open("/post_process/predictions.txt", "a+") as f:
                #         f.writelines(f"{video_id} {prediction[proposal_idx]} {start_frame_idxs[proposal_idx]} {end_frame_idxs[proposal_idx]}")
        