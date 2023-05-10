import torch
import torchinfo
import torchvision
from torchinfo import summary
from torchvision.models import vgg16, VGG16_Weights

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)

# Print a summary using torchinfo 
summary(model=model, 
        input_size=(32, 3, 224, 224), # (batch_size, 3 (RGB), 224 (H), 224 (W))
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
) 
