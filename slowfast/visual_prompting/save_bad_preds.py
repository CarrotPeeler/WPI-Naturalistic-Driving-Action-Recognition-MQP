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

import prompters
from utils import AverageMeter, ProgressMeter, save_checkpoint, cosine_lr, launch_job


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

    args.filename = '{}_{}_{}_lr_{}_decay_{}_trial_{}'. \
        format(args.method, args.prompt_size,
               args.optim, args.learning_rate, args.weight_decay, args.trial)

    args.model_folder = os.path.join(args.model_dir, args.filename)
    if not os.path.isdir(args.model_folder):
        os.makedirs(args.model_folder)

    args.image_folder = os.path.join(args.image_dir, args.filename)
    if not os.path.isdir(args.image_folder):
        os.makedirs(args.image_folder)

    return args

best_acc1 = 0
device = "cuda" if torch.cuda.is_available() else "cpu"

def main(args, cfg):
    global best_acc1, device

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # create model
    model = build_model(cfg)
    cu.load_test_checkpoint(cfg, model)
    model.eval()

    # create prompt
    prompter = prompters.__dict__[args.method](args).to(device)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            prompter.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # create dataloaders
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    test_loader = loader.construct_loader(cfg, "test")

    cudnn.benchmark = True

    for epoch in range(args.epochs):
        # remove zero-based indexing on epoch
        epoch += 1 

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, epoch)
        if hasattr(train_loader.dataset, "_set_epoch_num"):
            train_loader.dataset._set_epoch_num(epoch)

        # train for one epoch
        if(epoch == 1): 
            for batch_iter, (inputs, labels, index, times, meta) in enumerate(val_loader):
                if cfg.NUM_GPUS:
                    if isinstance(inputs, (list,)):
                        for i in range(len(inputs)):
                            if isinstance(inputs[i], (list,)):
                                for j in range(len(inputs[i])):
                                    inputs[i][j] = inputs[i][j].cuda(non_blocking=True)
                            else:
                                inputs[i] = inputs[i].cuda(non_blocking=True)
                    else:
                        inputs = inputs.cuda(non_blocking=True)
                    if not isinstance(labels, list):
                        labels = labels.cuda(non_blocking=True)

                images = inputs[0]
                images = images.to(device)
                labels = labels.tolist()

                batch_preds = model([images]).argmax(dim=1).tolist()

                for idx in range(len(batch_preds)):
                    if(batch_preds[idx] != labels[idx]):
                        clip = images[idx].permute(1, 0, 2, 3)
                        
                        for jdx, image in enumerate(clip):
                            if(jdx == 0):
                                save_image(image, os.getcwd() + f"/visual_prompting/bad_val_images/batch_{batch_iter}_clip_{idx}_prompt_{jdx}.png")
                            else: 
                                break



if __name__ == '__main__':

    args = parse_option()
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
        cfg = assert_and_infer_cfg(cfg)

    args.image_size = cfg.DATA.TRAIN_CROP_SIZE

    launch_job(cfg=cfg, args=args, init_method=args.init_method, func=main)