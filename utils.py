import glob
import os
import subprocess
import sys
import torch

def _get_frame_idxs_uniform(start_fidx, end_fidx, num_frames):
    """ sames as pyslowfast/datasets/decoder.py -> temporal_sampling
    """
    frame_idxs = []

    # get k float points uniformly within [s, e]
    index = torch.linspace(start_fidx, end_fidx, num_frames)
    # make sure frame_idxs all within range
    # long() is same as running int() on the float points
    index = torch.clamp(index, 0, 2000 - 1).long()

    frame_idxs = index.numpy().tolist()

    return frame_idxs

print(_get_frame_idxs_uniform(start_fidx=16, end_fidx=79, num_frames=16))