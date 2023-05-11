import os
import sys
sys.path.insert(1, os.getcwd())
from prepare_data import UCF101_Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models import VGG16_Weights
from torchinfo import summary
from train_functions import train_model
from models import VGG16_Mod
from torch import nn
from timeit import default_timer as timer 

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device} | Torch Version: {torch.__version__}")

torch.manual_seed(42)
torch.cuda.manual_seed(42)

######################################## LOADING THE DATA ############################################

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

num_classes = train_data.num_classes()

####################################### RUNNING THE MODEL ######################################

# Load the model
model = VGG16_Mod(num_classes, device)

# Print a summary using torchinfo 
summary(model=model, 
        input_size=(32, 3, 224, 224), # (batch_size, 3 (RGB), 224 (H), 224 (W))
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
) 

# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

start_time = timer()
results = train_model(model=model,
                      train_dataloader=train_dataloader,
                      test_dataloader=test_dataloader,
                      loss_fn=loss_fn,
                      optimizer=optimizer,
                      epochs=200,
                      device=device)
end_time = timer()

total_time = end_time - start_time

print(f"Total Training Time: {total_time:.2f} seconds")
