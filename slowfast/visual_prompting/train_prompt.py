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
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=0.2,
                        help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=1e-3,
                        help="weight decay")
    parser.add_argument("--warmup", type=int, default=30,
                        help="number of steps to warmup for")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--patience', type=int, default=1000)

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

    # define criterion and optimizer
    optimizer = torch.optim.SGD(prompter.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    total_steps = len(train_loader) * args.epochs
    scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup, total_steps)

    cudnn.benchmark = True

    if args.evaluate:
        acc1 = validate(val_loader, model, prompter, criterion, args)
        return

    epochs_since_improvement = 0

    for epoch in range(args.epochs):
        # remove zero-based indexing on epoch
        epoch += 1 

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, epoch)
        if hasattr(train_loader.dataset, "_set_epoch_num"):
            train_loader.dataset._set_epoch_num(epoch)

        # train for one epoch
        train(train_loader, model, prompter, optimizer, scheduler, criterion, epoch, args, cfg)
       
        # evaluate on validation set
        acc1 = validate(val_loader, model, prompter, criterion, args, cfg)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if(epoch % args.save_freq == 0 and du.get_rank() == 0):
            save_checkpoint({
                'epoch': epoch,
                'state_dict': prompter.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, args, is_best=is_best)

        if is_best:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            print(f"There's no improvement for {epochs_since_improvement} epochs.")

            if epochs_since_improvement >= args.patience:
                print("The training halted by early stopping criterion.")
                break



def train(train_loader, model, prompter, optimizer, scheduler, criterion, epoch, args, cfg):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix=f"Epoch: [{epoch}/{args.epochs}]")

    # switch to train mode
    prompter.train()

    num_batches_per_epoch = len(train_loader)

    end = time.time()
    for batch_iter, (inputs, labels, index, times, meta) in enumerate(train_loader):
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
        target = labels

        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate
        step = num_batches_per_epoch * epoch + batch_iter
        scheduler(step)

        images = images.to(device)
        target = target.to(device)

        prompted_images = prompter(images)

        output = model(prompted_images)

        # save prompted_images for visualization
        if((epoch == 1 or epoch % args.save_freq/2 == 0) and batch_iter == 0):
            for idx in range(len(prompted_images[0])): 
                # clip = images[idx].permute(1, 0, 2, 3) # non-prompted clip
                prompted_clip = prompted_images[0][idx].permute(1, 0, 2, 3) # prompted clip

                for jdx in range(prompted_clip.shape[0]):
                    if(jdx == 0):
                        # save_image(clip[jdx], os.getcwd() + f"/visual_prompting/images/originals/epoch_{epoch}_batch_{batch_iter}_clip_{idx}.png")
                        save_image(prompted_clip[jdx], f"{args.image_folder}/epoch_{epoch}_batch_{batch_iter}_clip_{idx}.png")
                    else: 
                        break
    
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy
        acc1 = topk_accuracies(output, target, (1,))[0]

        if cfg.NUM_GPUS > 1:
            acc1 = du.all_reduce([acc1])

        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_iter % args.print_freq == 0 and du.get_rank() == 0:
            progress.display(batch_iter)

        # if i % args.save_freq == 0:
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': prompter.state_dict(),
        #         'best_acc1': best_acc1,
        #         'optimizer': optimizer.state_dict(),
        #     }, args)
        
        torch.cuda.synchronize()
   
    del inputs

    # in case of fragmented memory
    torch.cuda.empty_cache()
   
    return losses.avg, top1.avg


def validate(val_loader, model, prompter, criterion, args, cfg):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1_org = AverageMeter('Original Acc@1', ':6.2f')
    top1_prompt = AverageMeter('Prompt Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1_org, top1_prompt],
        prefix='Validate: ')

    # switch to evaluation mode
    prompter.eval()

    with torch.no_grad():
        end = time.time()
        for batch_iter, (inputs, labels, index, times, meta) in enumerate(val_loader):
            if cfg.NUM_GPUS:
                # Transferthe data to the current GPU device.
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda()

            images = inputs[0]
            target = labels

            images = images.to(device)
            target = target.to(device)
            prompted_images = prompter(images)

            # compute output
            output_prompt = model(prompted_images)
            output_org = model(inputs)

            loss = criterion(output_prompt, target)

            # measure accuracy and record loss
            acc1_org = topk_accuracies(output_org, target, (1,))[0]
            acc1_prompt = topk_accuracies(output_prompt, target, (1,))[0]

            if cfg.NUM_GPUS > 1:
                acc1_org, acc1_prompt = du.all_reduce([acc1_org, acc1_prompt])
                                                                            
            losses.update(loss.item(), images.size(0))
            top1_org.update(acc1_org.item(), images.size(0))
            top1_prompt.update(acc1_prompt.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_iter % args.print_freq == 0 and du.get_rank() == 0:
                progress.display(batch_iter)

        if(du.get_rank() == 0): # only print this on 1 GPU
            disp_text = 'FINAL * Prompt Acc@1 {top1_prompt.avg:.3f} Original Acc@1 {top1_org.avg:.3f}'.format(top1_prompt=top1_prompt, top1_org=top1_org)
            print(disp_text)
            # write to log file as well
            with open(os.getcwd() + "/visual_prompting/prompt_train.log", "a+") as f:
                f.writelines(disp_text + "\n")

    return top1_prompt.avg


if __name__ == '__main__':

    args = parse_option()
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
        cfg = assert_and_infer_cfg(cfg)

    args.image_size = cfg.DATA.TRAIN_CROP_SIZE

    launch_job(cfg=cfg, args=args, init_method=args.init_method, func=main)