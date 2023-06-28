# FOR PRINTING OUT SIDE-BY-SIDE IMAGE TRANSFORMATION COMPARISONS FROM DATA LOADER



# Run command:
# cd slowfast
# python3 evaluation/print_loader.py --cfg configs/MVITv2_B_32x3_inf.yaml




from __future__ import print_function

import argparse
import os
import random

import torch
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image

from slowfast.datasets import loader
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.parser import load_config
from slowfast.datasets import transform

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

    args = parser.parse_args()

    return args

device = "cuda" if torch.cuda.is_available() else "cpu"

def main(args, cfg):
    global device

    # create dataloaders
    # train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    # test_loader = loader.construct_loader(cfg, "test")

    lder = val_loader

    for batch_iter, data in enumerate(lder):
        if(cfg.DATA.CROP_PROMPT == True and cfg.DATA.RETURN_CROPPING_PARAMS == True):
            inputs, labels, index, times, meta, crop_params_dict = data

            aspect_ratio_0_batch = crop_params_dict["aspect_ratio"][0].tolist()
            aspect_ratio_1_batch = crop_params_dict["aspect_ratio"][1].tolist()

            scale_0_batch = crop_params_dict["scale"][0].tolist()
            scale_1_batch = crop_params_dict["scale"][1].tolist()
        else:
            inputs, labels, index, times, meta = data
        
        images = inputs[0]
        
        if(batch_iter <= 5):
            for idx in range(images.shape[0]):
                if(cfg.DATA.CROP_PROMPT == True and cfg.DATA.RETURN_CROPPING_PARAMS == True):
                    frames_crop_train = transform.random_resized_crop(
                        images=images[idx],
                        target_height=crop_params_dict["crop_size"][idx].item(),
                        target_width=crop_params_dict["crop_size"][idx].item(),
                        scale=(scale_0_batch[idx], scale_1_batch[idx]),
                        ratio=(aspect_ratio_0_batch[idx], aspect_ratio_1_batch[idx]),
                    )

                    frames_crop_jit, _ = transform.random_short_side_scale_jitter(
                        images=images[idx],
                        min_size=crop_params_dict["min_scale"][idx].item(),
                        max_size=crop_params_dict["max_scale"][idx].item(),
                        inverse_uniform_sampling=crop_params_dict["inverse_uniform_sampling"][idx].item(),
                    )

                    frames_crop_val, _ = transform.random_crop(frames_crop_jit, crop_params_dict["crop_size"][idx].item())
                    frames_crop_test, _ = transform.uniform_crop(frames_crop_jit, crop_params_dict["crop_size"][idx].item(), spatial_idx=1)
                    frames_orig = images[idx]
                    
                    clip_crop_train = frames_crop_train.permute(1, 0, 2, 3)
                    clip_crop_val = frames_crop_val.permute(1, 0, 2, 3)
                    clip_crop_test = frames_crop_test.permute(1, 0, 2, 3)

                    clip_orig = torch.nn.functional.interpolate(
                        frames_orig.permute(1, 0, 2, 3),
                        size=(crop_params_dict["crop_size"][idx].item(), crop_params_dict["crop_size"][idx].item()),
                        mode="bilinear",
                        align_corners=False,
                    )

                    clip = torch.cat([clip_orig, clip_crop_train, clip_crop_val, clip_crop_test], dim=3)

                else:
                    frames = images[idx]
                    clip = frames.permute(1, 0, 2, 3)
            
                for jdx in range(clip.shape[0]):
                    if(jdx == 0):
                        save_image(clip_orig[jdx], os.getcwd() + f"/visual_prompting/images/originals/{batch_iter}_{idx}_{jdx}.png")


if __name__ == '__main__':

    # parse config and params
    args = parse_option()
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
        cfg = assert_and_infer_cfg(cfg)

    args.image_size = cfg.DATA.TRAIN_CROP_SIZE
    cfg.DATA.CROP_PROMPT = True
    cfg.DATA.RETURN_CROPPING_PARAMS = True

    # gather preds and targets from validation dataset
    launch_job(cfg=cfg, args=args, init_method=args.init_method, func=main)


