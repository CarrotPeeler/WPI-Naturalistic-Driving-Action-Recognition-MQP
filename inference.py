import os
import torch
import mmcv
import slowfast.utils.checkpoint as cu
from slowfast.models import build_model
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import List

class VideoProposalDataset(torch.utils.data.Dataset):
    def __init__(self, video_path, frame_length, frame_stride, proposal_stride, transform=None):
        self.video_path = video_path
        self.frame_length = frame_length
        self.transform = transform
        self.video = mmcv.VideoReader(video_path)

        proposal_length = frame_length * frame_stride

        # list of proposal tuples (start_frame_idx, end_frame_idx)
        self.proposals = self.generate_proposals(proposal_stride, proposal_length, len(self.video))
        
    """
    Returns list of numpy arrays (frames) given a list of the frame indices
    """
    def load_frames_by_idxs(self, idxs: List[int]):
        return [self.video[i] for i in idxs]
    
    """
    Returns list of proposal tuples (start_frame_idx, end_frame_idx)
    """
    def generate_proposals(self, proposal_stride, proposal_length, video_frame_count):
        proposals = []
        for i in range(0, video_frame_count, proposal_stride):
            # each proposal is a tuple with a start and end index
            proposals.append((i, i+proposal_length-1))
        return proposals
    
    """
    Returns a list of evenly spaced frame indices given a proposal and the number of frames to sample
    """
    def sample_frames(self, proposal, frame_length):
        return torch.linspace(proposal[0], proposal[1], frame_length, dtype=torch.int16).numpy().tolist()
        
    def __len__(self):
        "Returns the total number of proposals for this video."
        return len(self.proposals)

    # Returns tuple w/ img tensor and class: (torch.Tensor, int)
    def __getitem__(self, index: int):
        "Returns one proposal (frames, start_frame_idx, end_frame_idx)."
        proposal = self.proposals[index]
        sampled_frames_idxs = self.sample_frames(proposal, self.frame_length)
        
        # sampled_frames = self.load_frames_by_idxs(sampled_frames_idxs)

        # perform transform on frames if specified
        if self.transform:
            return (list(map(self.transform, sampled_frames_idxs)), proposal[0], proposal[1])
        else:
            return (sampled_frames_idxs, proposal[0], proposal[1])
        


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
    path_to_config = "/home/vislab-001/Jared/Naturalistic-Driving-Action-Recognition-MQP/slowfast/configs/SLOWFAST_8x8_R50.yaml"
    A2_data_path = "/home/vislab-001/Jared/SET-A2"
    frame_length = 16
    frame_stride = 4
    proposal_stride = 16 
    transform = None
    num_workers = os.cpu_count()
    batch_size = 1

    video_paths = glob(A2_data_path + "/**/*.MP4")
    args = parse_args()
    cfg = load_config(args, path_to_config)
    cfg = assert_and_infer_cfg(cfg)

    model = load_model(cfg)

    for i in tqdm(range(len(video_paths))):
        proposals_dataset = VideoProposalDataset(video_paths[i], frame_length, frame_stride, proposal_stride, transform)
        proposals_dataloader = DataLoader(dataset=proposals_dataset, batch_size=batch_size, num_workers=num_workers)

        model.eval()
        with torch.inference_mode():
            for batch_idx, (batch_frames, start_frame_idxs, end_frame_idxs) in enumerate(proposals_dataloader):
                # prediction = make_prediction(model, batch_frames)

                # # each batch has many proposals => we iterate through each proposal in a batch
                # for proposal_idx in enumerate(batch_frames.shape[0]):
                #     # write prob, start_frame_idx, end_frame_idx to file
                #     with open("/post_process/predictions.txt", "a+") as f:
                #         f.writelines(f"{prediction[proposal_idx]}, {start_frame_idxs[proposal_idx]}, {end_frame_idxs[proposal_idx]}")
        