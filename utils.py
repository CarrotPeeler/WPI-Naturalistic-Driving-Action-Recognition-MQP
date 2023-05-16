import os
import pathlib
import pandas as pd
import shutil
pd.options.mode.chained_assignment = None

from glob import glob
videos = glob("/home/vislab-001/Jared/SET-A1" + "/**/*.csv", recursive=True)
csv = pd.read_csv(videos[0])
row_data = csv.iloc[[0]]
path = os.getcwd() + "/trimmed_video_dump"
pathlib.Path(path).mkdir(parents=True, exist_ok=True)
shutil.rmtree(path, ignore_errors=True)


