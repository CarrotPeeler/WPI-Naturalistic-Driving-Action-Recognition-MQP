#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""


def add_custom_config(_C):
    # Add your own customized configs.

    # enable crop prompting for data loaders
    _C.DATA.CROP_PROMPT = False

    # enable to have data loaders return cropping parameters 
    _C.DATA.RETURN_CROPPING_PARAMS = False

    _C.DATA.CAM_VIEWS_METHODS = ['crop', 'noise_crop']

    _C.PROMPT.ENABLE = False

    _C.PROMPT.METHOD = ['padding', 'random_patch', 'fixed_patch']

    _C.PROMPT.PROMPT_SIZE = 224

    _C.PROMPT.RESUME = "./visual_prompting/save/models"

    _C.PROMPT.GPU = None

    _C.PROMPT.START_EPOCH = 1

    _C.PROMPT.LEARNING_RATE = 0.2

    _C.PROMPT.MOMENTUM = 0.9

    _C.PROMPT.WEIGHT_DECAY = 1e-3

    _C.PROMPT.WARM_UP = 30
