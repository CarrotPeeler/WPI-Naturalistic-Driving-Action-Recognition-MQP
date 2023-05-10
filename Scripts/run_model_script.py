from prepare_data import loadData, UCF101_Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader

# Load the organized data from the file system into arrays
train_1_dir = os.getcwd() + "/train_1"
X,y = loadData(train_1_dir + "/train_1_annotation.csv", train_1_dir)

# create a train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.2, stratify=y)

# Convert string labels for each img into array of width equal to num of categories
# Each row is an img => 0s and 1s to indicate which category image falls under
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

# Create UFC101_Dataset objects
train_data = UCF101_Dataset(X_train, y_train)
test_data = UCF101_Dataset(X_test, y_test)

# Setup the batch size hyper param
BATCH_SIZE = 32

# Turn dataset into iterables (batches)
train_dataloader = DataLoader(train_data,
                              BATCH_SIZE, 
                              shuffle=True)

test_dataloader = DataLoader(test_data,
                              BATCH_SIZE, 
                              shuffle=False)

print(f"Number of classes: {y_train.shape[1]}")