# Naturalistic-Driving-Action-Recognition-MQP

## Repository for Developing Video Classification/Action Recognition Models for the AI City Track Challenge #3 (2023)
 
Completed as part of my Major Qualifying Project for Worcester Polytechnic Institute (Summer 2023)

### Requirements
- Python >= 3.8.10, pandas, tqdm, scikit-learn, pytorch == 1.8.0+c111, tensorrt (may have to upgrade pip3 first)
- FFmpeg >= 4.2.7
- GNU Parallel 

### Setup
- Download the data for track #3 ['here'](https://www.aicitychallenge.org/2023-data-and-evaluation/)
- Download the checkpoint (SLOWFAST_8x8_R50) from the MODEL_ZOO.md and place file in checkpoints folder
- create an empty folder within the repo where the video clips will be dumped 
- in prepare_data.py, edit line 219 with the path of this empty folder and line 220 with the path of the A1 folder

### Data Preparation
- make sure you cd into this repo first, then run prepare_data.py after changing params as mentioned above

### Training
- cd into the slowfast folder and run the following:
`python3 tools/run_net.py --cfg configs/SLOWFAST_8x8_R50.yaml DATA.PATH_TO_DATA_DIR .`

### TODO
videosToClips:
- before splitting videos into clips using ffmpeg, 
    need to also check video length &
    create clips for empty durations where no distracted behavior happens and label them with -1 :heavy_check_mark:

- annotation file should only have video clip file path and label (format the file as csv but delimit path and label by a ' ') :heavy_check_mark:

- change model_train script to go back to using train_test_split on video clip data :o:

- set up PySlowFast with a Model and create config file to use for training (look at repos for examples) :o:

- create post-processing scripts to handle inference and temporal action localization output/accuracy :o:

- add data augmentation (color/flip images horizontally to add more data) :o:

- fixed crop for each camera view (rear, front, side) to only include driver area :o:
    OR 
    use action detection model to create BBoxes for var crop

- experiment with different models for action classification :o:

- incorporate visual prompting or experiment with other action recognition aspects :o:
