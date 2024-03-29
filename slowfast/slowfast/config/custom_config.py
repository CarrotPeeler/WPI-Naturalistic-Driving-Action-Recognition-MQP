#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""


from fvcore.common.config import CfgNode


def add_custom_config(_C):
    # Add your own customized configs.

    _C.TAL = CfgNode()
    _C.TAL.ENABLE = False
    
    # filters out prediction probabilities lower than this value
    _C.TAL.FILTERING_THRESHOLD = 0.9

    # Max number of clips to aggregate at once (MUST be a multiple of cfg.DATA.NUM_FRAMES)
    _C.TAL.CLIP_AGG_THRESHOLD = 64

    # percent of frames to evenly sample among the clip aggregation pool and the single proposal input
    _C.TAL.AGG_SAMPLING_RATIO = 0.5

    # clip threshold for re-evaluation of short segments (inclusive: <=)
    # for reference, 1 clip is equal to DATA.NUM_FRAMES
    _C.TAL.RE_EVAL_CLIP_THRESHOLD = 4

    _C.TAL.GAUSSIAN_SIGMA = 3

    _C.TAL.CANDIDATE_BONUS_SCORE_PER_SEC = 0.01

    # Enables the use of a 2nd GPU for TAL (speeds up inferencing)
    _C.TAL.USE_2_GPUS = False

    # output file path where results from TAL will be saved
    _C.TAL.OUTPUT_FILE_PATH = './inference/submission_files/sub_file.txt'

    _C.TAL.PRINT_DEBUG_OUTPUT = False

    # enable crop prompting for data loaders
    _C.DATA.CROP_PROMPT = False

    # enable to have data loaders return cropping parameters 
    _C.DATA.RETURN_CROPPING_PARAMS = False

    _C.DATA.CAM_VIEWS_METHODS = ['crop', 'noise_crop']

    _C.PROMPT = CfgNode()

    _C.PROMPT.ENABLE = False

    _C.PROMPT.METHOD = 'fixed_patch'

    _C.PROMPT.PROMPT_SIZE = 224

    _C.PROMPT.RESUME = None # "./visual_prompting/save/models/..."

    _C.PROMPT.GPU = None

    _C.PROMPT.START_EPOCH = 1

    _C.PROMPT.LEARNING_RATE = 0.2

    _C.PROMPT.MOMENTUM = 0.9

    _C.PROMPT.WEIGHT_DECAY = 1e-3

    _C.PROMPT.WARMUP = 30

    _C.PROMPT.IMAGE_FOLDER = './visual_prompting/save/images/mvitv2-b_fixed_patch'

    _C.PROMPT.PRINT_GRADS = False

    _C.PROMPT.MODEL_FOLDER = './visual_prompting/save/models/mvitv2-b_fixed_patch'

    _C.PROMPT.SELECTIVE_UPDATING = False
