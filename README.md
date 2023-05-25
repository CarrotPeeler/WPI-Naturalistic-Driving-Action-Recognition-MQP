# Naturalistic-Driving-Action-Recognition-MQP

## Repository for Developing Video Classification/Action Recognition Models for the AI City Track Challenge #3 (2023)
 
Done as part of my MQP for WPI (Summer 2023)


### TODO
videosToClips:
- before splitting videos into clips using ffmpeg, 
    need to also check video length &
    create clips for empty durations where no distracted behavior happens and label them with -1

    get GNU parallel working for FFmpeg using os.get_cpu()

- annotation file should only have video clip file path and label (format the file as csv but delimit path and label by a ' ')

- change model_train script to go back to using train_test_split on video clip data

- set up PySlowFast with a Model and create config file to use for training (look at repos for examples)

- create post-processing scripts to handle inference and temporal action localization output/accuracy

- add data augmentation (color/flip images horizontally to add more data)
- fixed crop for each camera view (rear, front, side) to only include driver area 
    OR 
    use action detection model to create BBoxes for var crop

- experiment with different models for action classification

- incorporate visual prompting or experiment with other action recognition aspects
