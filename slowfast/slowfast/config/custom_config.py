#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""


def add_custom_config(_C):
    # Add your own customized configs.

    # enable crop prompting for data loaders
    _C.DATA.CROP_PROMPT = False

    # enable to have data loaders return cropping parameters 
    _C.DATA.RETURN_CROPPING_PARAMS = False
