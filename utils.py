import os
import pandas as pd
pd.options.mode.chained_assignment = None

from glob import glob
csvs = glob("/home/vislab-001/Jared/SET-A1" + "/**/*.csv", recursive=True)

# TODO: convert timestamps column to frames
def convert_timestamps_to_frames(dataframe):
    pass

