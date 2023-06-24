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
# python3 evaluation/diagnose_preds.py --cfg configs/MVITv2_B_32x3_inf.yaml




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

    # create dataloaders
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    test_loader = loader.construct_loader(cfg, "test")

    lder = val_loader

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
            for batch_iter, (inputs, labels, index, times, meta) in enumerate(lder):
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

                output = model([images])
                max_tup = output.max(dim=1)
                batch_preds = max_tup[1]

                pred_labels.append(batch_preds.cpu())
                target_labels.append(labels.cpu())

                # the code below is for saving and examining incorrectly predicted input frames from clips

                batch_probs = max_tup[0].tolist()
                batch_preds, labels = batch_preds.tolist(), labels.tolist()

                for idx in range(len(batch_preds)):
                    clip_name = lder.dataset._path_to_videos[index[idx]].rpartition('/')[-1]

                    if(batch_preds[idx] != labels[idx] and save_incorrect_preds == True):
                        pass
                        with open(filepath, "a+") as f:
                            f.writelines(f"{clip_name},{class_dict[batch_preds[idx]]},{batch_probs[idx]:.3f},{class_dict[labels[idx]]}\n")
                    elif(batch_preds[idx] == labels[idx] and save_correct_preds == True):
                        with open(filepath_2, "a+") as f:
                            f.writelines(f"{clip_name},{class_dict[batch_preds[idx]]},{batch_probs[idx]:.3f},{class_dict[labels[idx]]}\n")
                            
                    if(batch_preds[idx] != labels[idx] and labels[idx] in classes_to_match and save_incorrect_image == True):
                        clip = images[idx].permute(1, 0, 2, 3)
                        for jdx in range(clip.shape[0]):
                            if(jdx == 3):
                                save_image(clip[jdx], os.getcwd() + f"/visual_prompting/images/bad_val_images_passenger_talk/{clip_name}_{jdx}_pred_{class_dict[batch_preds[idx]]}_prob_{batch_probs[idx]:.3f}_target_{class_dict[labels[idx]]}.png")
                    elif(batch_preds[idx] == labels[idx] and labels[idx] in classes_to_match and save_correct_image == True):
                        clip = images[idx].permute(1, 0, 2, 3)
                        for jdx in range(clip.shape[0]):
                            if(jdx == 3):
                                save_image(clip[jdx], os.getcwd() + f"/visual_prompting/images/good_val_images_passenger_talk/{clip_name}_{jdx}_pred_{class_dict[batch_preds[idx]]}_prob_{batch_probs[idx]:.3f}_target_{class_dict[labels[idx]]}.png")



if __name__ == '__main__':

    save_conf_mat = True
    save_incorrect_preds = True
    save_correct_preds = True
    save_incorrect_image = False
    save_correct_image = False

    classes_to_match = [11, 12]

    # parse config and params
    args = parse_option()
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
        cfg = assert_and_infer_cfg(cfg)

    args.image_size = cfg.DATA.TRAIN_CROP_SIZE
    cfg.TRAIN.BATCH_SIZE = 4

    # delete existing post_processed_data.txt if exists
    filepath = os.getcwd() + "/evaluation/val_preds/mvitv2-b_normal_data_no_rand_flip/incorrect_preds/" + "val_incorrect_pred_probs_mvitv2-b_180_epochs.txt"
    if os.path.exists(filepath):
        os.remove(filepath)

    filepath_2 = os.getcwd() + "/evaluation/val_preds/mvitv2-b_normal_data_no_rand_flip/correct_preds/" + "val_correct_pred_probs_mvitv2-b_180_epochs.txt"
    if os.path.exists(filepath_2):
        os.remove(filepath_2)

    # retrieve class names dict
    class_dict = getClassNamesDict(os.getcwd().rpartition('/')[0] + "/rq_class_names.txt")

    pred_labels = []
    target_labels = []

    # gather preds and targets from validation dataset
    launch_job(cfg=cfg, args=args, init_method=args.init_method, func=main)

    # concat list pred and target tensors to create tensors of preds and targets
    pred_labels_tensor, target_labels_tensor = torch.cat(pred_labels), torch.cat(target_labels)

    # create confusion matrix
    confmat = ConfusionMatrix(task='multiclass',num_classes=len(class_dict))
    confmat_tensor = confmat(preds=pred_labels_tensor,
                            target=target_labels_tensor)

    # wrap class names since they are rather long
    class_names = [ '\n'.join(wrap(l, 20)) for l in list(class_dict.values())]

    # plot confusion matrix
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),
        class_names=class_names,
        figsize=(14,14)
    )
    
    fig.suptitle("Validation Confusion Matrix for MViTv2-B", fontsize=15)
    
    # save figure
    if(save_conf_mat == True):
        fig.savefig(os.getcwd() + "/evaluation/graphs/mvitv2-b_normal_data_16x4_no_rand_flip/val_confusion_matrix_mvitv2-b_normal_data_180_epochs.png") # save the figure to file
    
    print("Done diagnosing predictions")

