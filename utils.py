import os
import pandas as pd

from glob import glob
csvs = glob("/home/vislab-001/Jared/SET-A1" + "/**/*.csv", recursive=True)
df = pd.read_csv(csvs[0])
videos = glob(csvs[0].rpartition('/')[0] + "/*.MP4")

video_name = videos[0].replace("NoAudio_", "").rpartition('/')[-1].split('.')[0]
video_view = video_name.split('_')[0]
video_endnum = video_name.split('_')[-1]

video_start_rows = df.loc[df["Filename"].notnull()]
video_start_rows.reset_index(inplace=True)

video_index = video_start_rows.index[video_start_rows["Filename"].str.partition('_')[0] == video_view and
                                     video_start_rows["Filename"].str.rpartition('_')[-1] == video_endnum].to_list()

print(video_index)
#next_video_index = -1

# if video_index + 1 < len(video_start_rows):
#     next_video_index = video_start_rows.loc[[video_index + 1]]
#     print(next_video_index)

#print(df.loc[df.index[df['Filename'] == ]])