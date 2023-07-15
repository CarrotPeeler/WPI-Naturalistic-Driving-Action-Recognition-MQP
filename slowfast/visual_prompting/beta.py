import os
import torch
import torch.nn as nn
import numpy as np
from fvcore.common.config import CfgNode
from torchvision.utils import save_image


# max_size = 30
# randten = torch.randn([1, 3, 224, max_size*2])
# offset_right = int(np.random.randint(1, max_size+1)) 
# offset_left = max_size*2 - offset_right

# pad_left = torch.nn.functional.interpolate(
#         randten,
#         size=(randten.shape[2], offset_left),
#         mode="bilinear",
#         align_corners=False,
#     )

# save_image(randten, os.getcwd() + f"/visual_prompting/images/self_attn/orig.png")
# save_image(pad_left, os.getcwd() + f"/visual_prompting/images/self_attn/interp.png")