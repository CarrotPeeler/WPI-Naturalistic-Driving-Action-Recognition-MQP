import os
import decord 
import pandas as pd
from glob import glob
from tqdm.auto import tqdm
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args

"""
Returns dict of video_ids and their corresponding video file names
"""
def get_video_ids_dict(path_to_csv):
    video_ids_dict = dict()
    df = pd.read_csv(path_to_csv)

    for idx, row in df.iterrows():
        key = int(row['video_id'])
        val = row[df.columns[1:4]].to_list()
        video_ids_dict[int(key)] = val

    return video_ids_dict



"""
Uniformly trims test data set videos into clips and generates a corresponding csv
"""
def uniform_video_segment(video_filepaths, video_extension, video_ids_dict, proposal_stride, proposal_length, clip_resolution, encode_speed, re_encode=False):
    # create save dir for clips if it doesn't exist
    clip_dir = os.getcwd().rpartition('/')[0] + "/data_loc"
    if(not os.path.exists(clip_dir)):
        os.mkdir(clip_dir)

    # delete existing test.csv if exists
    csv_path = os.getcwd() + "/test.csv"
    if os.path.exists(csv_path):
        os.remove(csv_path)

    video_paths = glob(video_filepaths + "/**/*" + video_extension)

    for i in tqdm(range(len(video_paths))):
        video_name = video_paths[i].rpartition('/')[2]

        video_id =  {i for i in video_ids_dict if video_name in video_ids_dict[i]}
        video_id = list(video_id)[0]
        
        # remove video extension
        video_name = video_name.rpartition('.')[0]

        frames = decord.VideoReader(video_paths[i])
        fps = int(frames.get_avg_fps())

        for j in range(0, len(frames) - proposal_stride - 1, proposal_stride):
            # generate proposal start and end frame index
            start_frame_idx = j
            end_frame_idx = j + proposal_length -1

            # calculate start and end times in format ss.ms (seconds, milliseconds)
            start_time = round(start_frame_idx / fps, 2)
            end_time = round(end_frame_idx / fps, 2)

            clip_filepath = os.getcwd().rpartition('/')[0] + f"/data_loc/{video_name}-{start_frame_idx}-{end_frame_idx}.MP4"

            if(encode_speed == "default"): 
                preset = ""
            else:
                preset = "-preset " + encode_speed + " "

            # write ffmpeg command to bash script
            with open(clip_dir + "/ffmpeg_loc_commands.sh", 'a+') as f:
                if(re_encode == False):
                    f.writelines(f"ffmpeg -loglevel quiet -y -i {video_paths[i]} -ss {start_time} -to {end_time} -c:v copy {clip_filepath}\n")
                else:
                    f.writelines(f"ffmpeg -loglevel quiet -y -i {video_paths[i]} -vf scale={clip_resolution} -ss {start_time} -to {end_time} -c:v libx264 {preset}{clip_filepath}\n")

            # add clip path, placeholder label, video_id, start_time, end_time to test.csv
            with open(os.getcwd() + "/test.csv", "a+") as f:
                f.writelines(f"{clip_filepath} 0 {video_id} {start_time} {end_time}\n")

    # parallelize ffmpeg commands
    os.system(f"parallel --eta < {clip_dir}/ffmpeg_loc_commands.sh")



# Always run the start method inside this if-statement
if __name__ == '__main__':  

    ############### CONFIGURATION PARAMS ################
    A2_data_path = "/home/vislab-001/Jared/SET-A2"
    clip_resolution = (512, 512) # resolution that clips will be resized to; this should match the resolution used in prepare_data.py
    proposal_stride = 64
    encode_speed = "ultrafast"
    clip_resolution="512:512"
    #####################################################

    args = parse_args()
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
        cfg = assert_and_infer_cfg(cfg)

    frame_length = cfg.DATA.NUM_FRAMES
    frame_stride = cfg.DATA.SAMPLING_RATE
    proposal_length = frame_length * frame_stride

    """
    Because of re-encoding, the videos may take several hours to process; to run script in the background:
    cd slowfast
    python3 prepare_loc_data.py < /dev/null > ffmpeg_log.txt 2>&1 &
    """
    # uniformly segment videos into clips of the same size 
    uniform_video_segment(video_filepaths=A2_data_path, 
                          video_extension=".MP4", 
                          video_ids_dict=get_video_ids_dict(os.getcwd() + "/inference/video_ids.csv"),
                          proposal_stride=proposal_stride, 
                          proposal_length=proposal_length, 
                          clip_resolution=clip_resolution, 
                          encode_speed=encode_speed,
                          re_encode=True)

    