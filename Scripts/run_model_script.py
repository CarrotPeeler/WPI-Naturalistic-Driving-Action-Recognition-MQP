import os
import sys
sys.path.insert(1, os.getcwd())
from prepare_data import UCF101_Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models import VGG16_Weights

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load the organized data from the file system into arrays
train_1_dir = os.getcwd() + "/train_1"
train_1_df = pd.read_csv(train_1_dir + "/train_1_annotation.csv")

# create a train-test split
train_df, test_df = train_test_split(train_1_df, random_state=42, test_size=.2, stratify=train_1_df['class'])

# Create UFC101_Dataset objects
train_data = UCF101_Dataset(train_df, img_dir=train_1_dir, transform=VGG16_Weights.IMAGENET1K_V1.transforms())
test_data = UCF101_Dataset(test_df, img_dir=train_1_dir, transform=VGG16_Weights.IMAGENET1K_V1.transforms())

# Setup the batch size hyper param
BATCH_SIZE = 32

# Turn dataset into iterables (batches)
train_dataloader = DataLoader(train_data,
                              BATCH_SIZE, 
                              shuffle=True)

test_dataloader = DataLoader(test_data,
                              BATCH_SIZE, 
                              shuffle=False)

print(f"Number of classes: {train_data.num_classes()}")
