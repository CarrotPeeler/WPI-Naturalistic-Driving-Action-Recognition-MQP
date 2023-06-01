import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
from glob import glob
from torchinfo import summary
import torch
from torch import nn
pd.options.mode.chained_assignment = None

print(torch.__version__)

print(torch.cuda.is_available())

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

force_cudnn_initialization()