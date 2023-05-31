import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
from glob import glob
from torchinfo import summary
import torch
pd.options.mode.chained_assignment = None

print(os.getcwd())