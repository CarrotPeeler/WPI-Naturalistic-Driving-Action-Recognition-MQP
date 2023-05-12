import torch
from torchvision.models import vgg16, VGG16_Weights, resnet18, ResNet18_Weights
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
    
# Model adapted from https://github.com/PacktPublishing/PyTorch-Computer-Vision-Cookbook
class Resnt18Rnn(nn.Module):
    def __init__(self, num_classes, dropout_rate, rnn_hidden_size, rnn_num_layers, batch_size):
        super(Resnt18Rnn, self).__init__()
        self.batch_size = batch_size
        
        baseModel = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_features = baseModel.fc.in_features
        baseModel.fc = Identity()
        self.baseModel = baseModel
        self.dropout= nn.Dropout(dropout_rate)
        self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers)
        self.fc1 = nn.Linear(rnn_hidden_size, num_classes)

    def forward(self, x):
        ii = 0
        y = self.baseModel((x[:,ii]))
        output, (hn, cn) = self.rnn(y.unsqueeze(1))
        for ii in range(1, self.batch_size):
            y = self.baseModel((x[:,ii]))
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
        out = self.dropout(out[:,-1])
        out = self.fc1(out) 
        return out 
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x 