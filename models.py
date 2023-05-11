import torch
from torchvision.models import vgg16, VGG16_Weights
from torch import nn

class VGG16_Mod(nn.Module):
    def __init__(self, output_shape, device) -> None:
        super().__init__()
        self.device = device
        
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        for param in model.features.parameters():
            param.requires_grad = False

        # mod the last classifier layer of VGG16
        model.classifier = torch.nn.Sequential(
            # output from the layer before (1280 units) = input units
            torch.nn.Linear(in_features=1280, out_features=640), 
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5, inplace=True), 
            
            torch.nn.Linear(in_features=640, out_features=320), 
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5, inplace=True), 

            torch.nn.Linear(in_features=160, out_features=80), 
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5, inplace=True), 

            torch.nn.Linear(in_features=80, out_features=40), 
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5, inplace=True), 

            torch.nn.Linear(in_features=40, out_features=self.output_shape, # num of classes
                            )).to(self.device)
        
        self.model = torch.compile(model)

    def forward(self, X):
        return self.model(X)
    