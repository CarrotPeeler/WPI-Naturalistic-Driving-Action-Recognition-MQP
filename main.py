import pandas as pd
from sklearn.model_selection import train_test_split
from prepare_data import *
import os

# Main Code Execution
filter = ['apply','blowdry','haircut','brushing','typing']

train = createDataFrame('/content/ucfTrainTestlist/trainlist01.txt')
test = createDataFrame('/content/ucfTrainTestlist/testlist01.txt')

filterDataFrame(train,filter)
filterDataFrame(test,filter)

train_1_dir = os.getcwd() + "/train_1"

videoToFrames(train, "~/Jared/ucf101", train_1_dir)
create_annotation(train_1_dir, train_1_dir, "train_1_annotation.csv")
X,y = loadData("train_1_annotation.csv", train_1_dir)

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