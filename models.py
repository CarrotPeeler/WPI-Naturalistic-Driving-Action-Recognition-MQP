import torch
from torchvision.models import vit_b_32, ViT_B_32_Weights
from torch import nn

class ViT_CLIP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.heads = nn.Sequential(
            nn.Linear(in_features=768, out_features=self.num_classes)
        )
        
    def forward(self, X):
        return self.model(X)