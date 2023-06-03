import glob
import os
import subprocess
import sys
import torch
import mmcv
from typing import List

tensor = torch.tensor([[1,1,1,1],
                       [2,2,2,2],
                       [3,3,3,3]])

for row in tensor:
    print(row.tolist())