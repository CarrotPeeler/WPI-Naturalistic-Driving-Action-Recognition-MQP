"""
MIT License

Copyright (c) 2022 Hyojin Bahng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""



# THIS FILE CREATES CONFUSION MATRICES, INCORRECT PRED INFO CSVS, AND SAVES INCORRECT PRED IMAGES



# Run command:
# cd slowfast
# python3 evaluation/print_loader.py --cfg configs/MVITv2_B_32x3_inf.yaml




from __future__ import print_function

import argparse
import os
from tqdm import tqdm
import time
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.parser import load_config
from slowfast.utils.metrics import topk_accuracies
from slowfast.datasets import transform
from prepare_data import getClassNamesDict

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from textwrap import wrap

from visual_prompting.utils import launch_job


def parse_option():

    parser = argparse.ArgumentParser('Visual Prompting for Vision Models')

    # pyslowfast cfg
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_files",
        help="Path to the config files",
        default=["configs/Kinetics/SLOWFAST_4x16_R50.yaml"],
        nargs="+",
    )
    parser.add_argument(
        "--opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )

    # visual prompting

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='save frequency')
    parser.add_argument('--epochs', type=int, default=400,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=40,
                        help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=1e-3,
                        help="weight decay")
    parser.add_argument("--warmup", type=int, default=30,
                        help="number of steps to warmup for")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--patience', type=int, default=10)

    # model
    parser.add_argument('--method', type=str, default='padding',
                        choices=['padding', 'random_patch', 'fixed_patch'],
                        help='choose visual prompting method')
    parser.add_argument('--prompt_size', type=int, default=30,
                        help='size for visual prompts')

    # other
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for initializing training')
    parser.add_argument('--model_dir', type=str, default='./visual_prompting/save/models',
                        help='path to save models')
    parser.add_argument('--image_dir', type=str, default='./visual_prompting/save/images',
                        help='path to save images')
    parser.add_argument('--filename', type=str, default=None,
                        help='filename to save')
    parser.add_argument('--trial', type=int, default=1,
                        help='number of trials')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to resume from checkpoint')
    parser.add_argument('--evaluate', default=False,
                        action="store_true",
                        help='evaluate model test set')
    parser.add_argument('--gpu', type=int, default=None,
                        help='gpu to use')

    args = parser.parse_args()

    return args

device = "cuda" if torch.cuda.is_available() else "cpu"

def main(args, cfg):
    global device

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # create dataloaders
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    test_loader = loader.construct_loader(cfg, "test")

    lder = train_loader

    for epoch in range(args.epochs):
        # remove zero-based indexing on epoch
        epoch += 1 

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, epoch)
        if hasattr(train_loader.dataset, "_set_epoch_num"):
            train_loader.dataset._set_epoch_num(epoch)

        # train for one epoch
        if(epoch == 1): 
            for batch_iter, (inputs, labels, index, times, meta) in enumerate(lder):

                images = inputs[0]

                scl, asp = (
                    cfg.DATA.TRAIN_JITTER_SCALES_RELATIVE,
                    cfg.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
                )
                relative_scales = (
                    scl
                )
                relative_aspect = (
                    asp
                )

                transform_func = transform.random_resized_crop

                if(batch_iter <= 5):
                    for idx in range(images.shape[0]):
                        frames_crop = transform_func(
                            images=images[idx],
                            target_height=256,
                            target_width=256,
                            scale=relative_scales,
                            ratio=relative_aspect,
                        )
                        frames_orig = images[idx]
                        
                        clip_crop = frames_crop.permute(1, 0, 2, 3)
                        clip_orig = frames_orig.permute(1, 0, 2, 3)

                        clip = torch.cat([clip_orig, clip_crop], dim=3)
                        for jdx in range(clip.shape[0]):
                            if(jdx == 0):
                                save_image(clip[jdx], os.getcwd() + f"/visual_prompting/images/side_by_sides/{batch_iter}_{idx}_{jdx}.png")


if __name__ == '__main__':

    # parse config and params
    args = parse_option()
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
        cfg = assert_and_infer_cfg(cfg)

    args.image_size = cfg.DATA.TRAIN_CROP_SIZE
    cfg.DATA.CROP_PROMPT = True

    # gather preds and targets from validation dataset
    launch_job(cfg=cfg, args=args, init_method=args.init_method, func=main)


