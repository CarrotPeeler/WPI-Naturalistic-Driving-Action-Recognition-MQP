import torch
from torchvision.models import vgg16, VGG16_Weights
from torch import nn

class VGG16_Mod(nn.Module):
    def __init__(self, output_shape) -> None:
        super().__init__()
        self.output_shape = output_shape
        
        self.model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        for param in self.model.features.parameters():
            param.requires_grad = False

        # mod the last classifier layer of VGG16
        self.model.classifier = torch.nn.Sequential(
            # output from the layer before (25088 units, found from summary()) = input units
            torch.nn.Linear(in_features=25088, out_features=1024), 
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5), 
            
            torch.nn.Linear(in_features=1024, out_features=512), 
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5), 

            torch.nn.Linear(in_features=512, out_features=256), 
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5), 

            torch.nn.Linear(in_features=256, out_features=128), 
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5), 

            torch.nn.Linear(in_features=128, out_features=self.output_shape)) # num of classes

    def forward(self, X):
        return self.model(X)
    