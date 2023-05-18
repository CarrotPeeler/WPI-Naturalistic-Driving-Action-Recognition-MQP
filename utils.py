import os
import pathlib
import pandas as pd
import shutil
from glob import glob
from torchinfo import summary
from torchvision.models import vit_b_32, ViT_B_32_Weights
pd.options.mode.chained_assignment = None

# Print a summary using torchinfo 
summary(model=vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1), 
        input_size=(32, 3, 224, 224), # (batch_size, 3 (RGB), 224 (H), 224 (W))
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
) 