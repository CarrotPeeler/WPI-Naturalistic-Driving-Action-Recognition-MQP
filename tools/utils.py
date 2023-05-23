import os
import pathlib
import pandas as pd
import shutil
import torch
from glob import glob
from torchinfo import summary
from torchvision.models import vit_b_32, ViT_B_32_Weights
pd.options.mode.chained_assignment = None

a = torch.linspace(0,15,16)
print(a)

