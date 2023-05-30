# Naturalistic-Driving-Action-Recognition-MQP

## Repository for Developing Video Classification/Action Recognition Models for the AI City Track Challenge #3 (2023)
 
Done as part of my MQP for WPI (Summer 2023)

### Requirements
- Python >= 3.8.10, pandas, tqdm, scikit-learn, pytorch >= 2.0.1
- FFmpeg >= 4.2.7
- GNU Parallel 

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
