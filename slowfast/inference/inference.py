import os
import torch
import decord
import pandas as pd
import slowfast.utils.checkpoint as cu
from slowfast.models import build_model
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import List


class VideoProposalDataset(torch.utils.data.Dataset):
    def __init__(self, video_path, frame_length, frame_stride, proposal_stride, transform=None, num_workers=1):
        self.video_path = video_path
        self.frame_length = frame_length
        self.transform = transform
        self.proposals = []
        
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
        
    def __len__(self):
        "Returns the total number of proposals for this video."
        return len(self.proposals)

    # Returns tuple w/ img tensor and class: (torch.Tensor, int)
    def __getitem__(self, index: int):
        "Returns one proposal (frames, start_frame_idx, end_frame_idx)."
        proposal = self.proposals[index]
    
        # perform transform on frames if specified
        if self.transform:
            return (list(map(self.transform, proposal[0])), proposal[1], proposal[2])
        else:
            return proposal
        


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
    path_to_config = "/home/vislab-001/Jared/Naturalistic-Driving-Action-Recognition-MQP/slowfast/configs/SLOWFAST_8x8_R50.yaml"
    A2_data_path = "/home/vislab-001/Jared/SET-A2"
    frame_length = 16
    frame_stride = 4
    proposal_stride = frame_length * frame_stride # for non-overlapping proposals; set smaller num for overlapping 
    transform = None
    num_workers = 8 #os.cpu_count()
    batch_size = 1
    ############################################

    video_ids_dict = get_video_ids_dict(os.getcwd() + "/inference/video_ids.csv")
    video_paths = glob(A2_data_path + "/**/*.MP4")
    args = parse_args()
    cfg = load_config(args, path_to_config)
    cfg = assert_and_infer_cfg(cfg)

    model = load_model(cfg)

    print(f"Number of threads: {num_workers}")

    for i in tqdm(range(len(video_paths))):
        proposals_dataset = VideoProposalDataset(video_paths[i], frame_length, frame_stride, proposal_stride, transform, num_workers=num_workers)
        proposals_dataloader = DataLoader(dataset=proposals_dataset, batch_size=batch_size, num_workers=num_workers)

        video_name = video_paths[i].rpartition('/')[2]
        video_id =  {i for i in video_ids_dict if video_name in video_ids_dict[i]}
        video_id = list(video_id)[0]

        model.eval()
        with torch.inference_mode():
            for batch_idx, (batch_frames, start_frame_idxs, end_frame_idxs) in enumerate(proposals_dataloader):
                prediction = make_prediction(model, batch_frames)

                # each batch has many proposals => we iterate through each proposal in a batch
                for proposal_idx in enumerate(batch_frames.shape[0]):
                    # write prob, start_frame_idx, end_frame_idx to file
                    with open("/post_process/predictions.txt", "a+") as f:
                        f.writelines(f"{video_id} {prediction[proposal_idx]} {start_frame_idxs[proposal_idx]} {end_frame_idxs[proposal_idx]}")
        