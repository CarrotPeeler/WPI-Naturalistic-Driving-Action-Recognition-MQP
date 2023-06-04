# Naturalistic-Driving-Action-Recognition-MQP

## Repository for Developing Video Classification/Action Recognition Models for the AI City Track Challenge #3 (2023)
 
Completed as part of my Major Qualifying Project for Worcester Polytechnic Institute (Summer 2023)

### Requirements
- Python >= 3.8.10, pytorch == 1.13.1+c117, pandas, tqdm, scikit-learn, decord, tensorrt (may have to upgrade pip3 first)
- FFmpeg >= 4.2.7 
- GNU Parallel 
- Follow PySlowFast INSTALL.md 

For video and decoder functionality to work, torchvision MUST be compiled & built from source (used v0.14.1+c117): 
- uninstall FFmpeg if you already have it, then reinstall it with the following command:
```console
apt install ffmpeg libavutil-dev libavcodec-dev libavformat-dev libavdevice-dev libavfilter-dev libswscale-dev libswresample-dev libswresample-dev libpostproc-dev libjpeg-dev libpng-dev
```
- clone the torchvision release compatible with your pytorch version
- add this line to top of setup.py: 
```python
sys.path.append("/home/vislab-001/.local/lib/python3.8/site-packages")
```
- to make sure the setup.py has full permissions, use the following command:
```console
sudo chmod 777 {path to torchvision repo}
```
- run the setup.py, if there are more permission errors, simply chmod 777 the folder location indicated by the errors


### Setup
- Download the data for track #3 ['here'](https://www.aicitychallenge.org/2023-data-and-evaluation/)
- Download the checkpoint (SLOWFAST_8x8_R50) from the MODEL_ZOO.md and place file in checkpoints folder (must edit the checkpoint path in slowfast config)
- create an empty folder within the repo where the video clips will be dumped 
- in prepare_data.py, edit the following to where you saved the A1 data folder and where you created the empty folder
```python
videos_loadpath = "/path_to_A1/SET-A1"
clips_savepath = "/path_to_data/data_dir"
```

### Data Preparation
- make sure you cd into this repo first, then run prepare_data.py after changing params as mentioned above

### Training
- edit the config in slowfast/slowfast/configs
- cd into outermost slowfast folder and run the following:
```console
python3 tools/run_net.py --cfg configs/SLOWFAST_8x8_R50.yaml DATA.PATH_TO_DATA_DIR .
```

### Inference
- cd into outermost slowfast folder (make you cd while within the python interpreter, not from bash)
- go to inference folder > run inference.py 

### TODO
- before splitting videos into clips using ffmpeg, 
    need to also check video length &
    create clips for empty durations where no distracted behavior happens and label them with -1 :heavy_check_mark:

- annotation file should only have video clip file path and label (format the file as csv but delimit path and label by a ' ') :heavy_check_mark:

- set up PySlowFast with a Model and create config file to use for training (look at repos for examples) :heavy_check_mark:

- setup config + model to use 2 gpus (breaks when attemtping) and more workers (max = 20 cpu threads); then, can increase train batch_size :o:

- edit eval output to show train and val accuracy and specify what top1 and top5 error apply to (train or val) :o:

- create proposal generation and post-processing scripts to handle inference on A2 and temporal action localization output/accuracy: :o:
    - create video_proposals_dataset(video_path, frame_length, frame_stride, proposal_stride, etc. params) (torch.dataset) class
        - for a single untrimmed video, use cv2 to convert video into frames, save frames to self.frames
        - proposal length = frame_length * frame_stride
        - def func to generate list of proposal tuples (start_idx, end_idx) given prop. length and prop. stride
        - def func to subsample {frame_length} # of frames, evenly spaced, for a single proposal; return list of subsampled frame idxs
        - self.proposals = list of proposals (each proposal is a list of subsampled frame idxs retrieved from func above)

        - for __get__item(): retrieve proposal (list) in self.proposals; return frames from self.frames using idxs in subsampled list 
        - __get__item() may also be modified to crop or perform other image transforms for later use
    
    - create ActionClassifier class that uses trained model to make inferences on given set of frames in a batch
        - for {frame_length} # of frames, average probs over all frames for a proposal, then take argmax to find class idx
        - return batch of idxs

    - inference script
        - for each untrimmed video path, create a video_proposals_dataset and dataloader for it
        - for each dataloader, feed batch of frames into model for predictions
        - for each proposal frame set in the batch, append (pred, start, end) to list of preds

    - post-processing script to piece together all proposals back into the full untrimmed video and align action preds with timestamps
    in the video; multiple proposals that are consecutive in temporal space, having the same action pred, should be combined into one start and end timestamp

- add data augmentation (color/flip images horizontally to add more data) :o:

- fixed crop for each camera view (rear, front, side) to only include driver area :o:
    OR 
    use action detection model to create BBoxes for var crop

- experiment with different models for action classification :o:

- incorporate visual prompting or experiment with other action recognition aspects :o:
