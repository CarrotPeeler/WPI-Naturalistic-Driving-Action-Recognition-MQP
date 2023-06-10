import os
import torch
import pandas as pd
import numpy as np
from glob import glob
from scipy import stats

# Always run the start method inside this if-statement
if __name__ == '__main__':  

    df = pd.read_csv(os.getcwd() + "/slowfast/train.csv", delimiter=" ", names=["path", "class"])
    print(df.pivot_table(index = ["class"], aggfunc = "size"))


#  # split data into train and val sets
#         splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state = 42)
#         split = splitter.split(X=df['clip'], y=df['class'])
#         train_indexes, test_indexes = next(split)

#         train_df = df.iloc[train_indexes]
#         test_df = df.iloc[test_indexes]

#         train_df.to_csv(os.getcwd() + "/slowfast/train.csv", sep=" ", header=False, index=False)
#         test_df.to_csv(os.getcwd() + "/slowfast/val.csv", sep=" ", header=False, index=False)