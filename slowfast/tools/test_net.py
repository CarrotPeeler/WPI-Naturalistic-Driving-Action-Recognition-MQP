#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch

from scipy import stats

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.env import pathmgr
from slowfast.utils.meters import AVAMeter, TestMeter
from visual_prompting import prompters

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None, prompter=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    if cfg.PROMPT.ENABLE == True and prompter is not None:
        prompter.eval()

    # delete existing predictions.txt if exists
    pred_output = os.getcwd() + "/post_process/predictions.txt"
    if os.path.exists(pred_output):
        print("DELETING predictions.txt")
        os.remove(pred_output)


    if cfg.TAL.ENABLE == True:
        activity_ids = []
        localization_tuples = []

        start_time = 0
        end_time = 0

        prev_agg_pred = -1

        # stores aggregated clips (batches) for each cam view type
        cam_view_clips = {}

        clip_agg_cnt = 0

        

    for cur_iter, (inputs, labels, video_idx, time, meta, proposal) in enumerate(
        test_loader
    ):

        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            # Transfer the data to the current GPU device.
            labels = labels.cuda()
            video_idx = video_idx.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        test_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = (
                ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
            )
            metadata = (
                metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
            )

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(preds, ori_boxes, metadata)
            test_meter.log_iter_stats(None, cur_iter)
        elif cfg.TASK == "ssl" and cfg.MODEL.MODEL_NAME == "ContrastiveModel":
            if not cfg.CONTRASTIVE.KNN_ON:
                test_meter.finalize_metrics()
                return test_meter
            # preds = model(inputs, video_idx, time)
            train_labels = (
                model.module.train_labels
                if hasattr(model, "module")
                else model.train_labels
            )
            yd, yi = model(inputs, video_idx, time)
            batchSize = yi.shape[0]
            K = yi.shape[1]
            C = cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM  # eg 400 for Kinetics400
            candidates = train_labels.view(1, -1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)
            retrieval_one_hot = torch.zeros((batchSize * K, C)).cuda()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(cfg.CONTRASTIVE.T).exp_()
            probs = torch.mul(
                retrieval_one_hot.view(batchSize, -1, C),
                yd_transform.view(batchSize, -1, 1),
            )
            preds = torch.sum(probs, 1)

        elif(cfg.PROMPT.ENABLE == True):

            if("multi_cam" in cfg.PROMPT.METHOD):
                cam_views = []
                for clip_idx in range(len(inputs[0])):
                    cam_view = test_loader.dataset._path_to_videos[video_idx[clip_idx]].rpartition('/')[-1].partition('_user')[0]
                    cam_views.append(cam_view)
                
                prompted_inputs = prompter(inputs[0], cam_views)
            else:
                prompted_inputs = prompter(inputs[0])
            
            preds = model(prompted_inputs)

        ######################## T A L ########################

        elif cfg.TAL.ENABLE == True and cfg.TEST.BATCH_SIZE == 3:
                
            if clip_agg_cnt == 0:
                # perform first iter setup
                cam_views = set()
                        
                # check that there's no duplicate cam views and video ids + start/end times are matching
                # also intialize cam view clips for storing aggregated clips 
                for i in range(cfg.TEST.BATCH_SIZE):
                    cam_view = test_loader.dataset._path_to_videos[video_idx[i]].rpartition('/')[-1].partition('_user')[0] 
                    cam_views.add(cam_view)
                    
                    cam_view_clips[cam_view] = inputs[0][i].permute(1,0,2,3)

                assert len(cam_views) == 3, f"Cam view mismatch for batch {cur_iter}"
                assert len(set(proposal[0])) == len(set(proposal[1])) == len(set(proposal[2])) == 1, f"Proposal mismatch for batch {cur_iter}"

                start_time = proposal[1][0]
                end_time = proposal[2][0]

            else:
                end_time = proposal[1][0]

                cviews = set()
                        
                # check that there's no duplicate cam views and video ids + start/end times are matching
                # Concat clips to their respective batch (based on cam view type) 
                for j in range(cfg.TEST.BATCH_SIZE):
                    cview = test_loader.dataset._path_to_videos[video_idx[j]].rpartition('/')[-1].partition('_user')[0]
                    cviews.add(cview)

                    cam_view_clips[cview] = torch.cat([cam_view_clips[cview], inputs[0][j].permute(1,0,2,3)], dim=0)
                    assert cam_view_clips[cview].shape[1] == 3 \
                        and cam_view_clips[cview].shape[2] == cfg.DATA.TEST_CROP_SIZE \
                        and cam_view_clips[cview].shape[3] == cfg.DATA.TEST_CROP_SIZE, f"Shape mismatch, got {cam_view_clips[cview].shape}"

                assert len(cviews) == 3, f"Cam view mismatch for next batch {cur_iter}"
                assert len(set(proposal[0])) == len(set(proposal[1])) == len(set(proposal[2])) == 1, f"Proposal mismatch for next batch {cur_iter}"

            # logger.info(f"CUR ITER: {cur_iter}")
                    
            all_cam_view_preds = []
            all_cam_view_probs = []

            for cam_view_type in cam_view_clips.keys():
                if(cam_view_type == "Dashboard"):
                    logger.info(f"NUM FRAMES AGGREGATED: {cam_view_clips[cam_view_type].shape[0]}")

                if cam_view_clips[cam_view_type].shape[0] == cfg.DATA.NUM_FRAMES:
                    input = [cam_view_clips[cam_view_type].permute(1,0,2,3).unsqueeze(dim=0)]
                
                else:
                    start_idx = 0
                    end_idx = start_idx + cam_view_clips[cam_view_type].shape[0] - 1
                    sampled = temporal_sampling(cam_view_clips[cam_view_type], start_idx, end_idx, cfg.DATA.NUM_FRAMES)
                    input = [sampled.permute(1,0,2,3).unsqueeze(dim=0)]

                cam_view_preds = model(input).cpu()
                
                labels = labels.cpu()
                video_idx = video_idx.cpu()

                cam_view_pred = cam_view_preds.argmax().item()
                cam_view_prob = cam_view_preds.max().item()

                # cam_view = test_loader.dataset._path_to_videos[video_idx[i]].rpartition('/')[-1].partition('_user')[0] 
                # logger.info(f"batch: {cur_iter}, {cam_view}, pred: {cam_view_pred}, prob: {cam_view_prob:.3f}")

                all_cam_view_preds.append(cam_view_pred)
                all_cam_view_probs.append(cam_view_prob)

            preds = np.array(all_cam_view_preds)
            probs = np.array(all_cam_view_probs)

            # check if there is a common pred among candidates
            if(stats.mode(preds, keepdims=False)[1] > 1):
                # cnt += 1
                # validate the probs of each pred are high enough to not be coincidence
                mode_pred = stats.mode(preds, keepdims=False)[0]

                mode_pred_idxs = np.where(preds == mode_pred)[0]
                minority_pred_idxs = np.where(preds != mode_pred)[0]

                mode_probs = np.array([probs[z] for z in mode_pred_idxs])
                minority_probs = np.array([probs[j] for j in minority_pred_idxs])

                # # check if minority pred exists and its prob is higher than mode probs
                # if(len(minority_pred_idxs) > 0 and minority_probs.max() > mode_probs.max() and mode_probs.max() < prob_threshold):
                #     # print(f"{preds} ==> {probs}")
                #     minority_pred = np.array([preds[m] for m in minority_pred_idxs]).max()
                #     agg_preds.append(minority_pred)
                #     agg_probs.append(minority_probs.max())
                
                # # select common pred if mean of common pred probs >= threshold
                # else:
                curr_agg_pred = mode_pred
                agg_prob = mode_probs.mean()
                logger.info(f"batch: {cur_iter}, agg pred: {curr_agg_pred}, agg prob: {agg_prob:.3f}")

            # no common pred => select highest prob pred among all three camera angles
            else:
                best_prob_idx = probs.argmax()
                curr_agg_pred = preds[best_prob_idx]
                agg_prob = probs.max()
                logger.info(f"batch: {cur_iter}, agg pred: {curr_agg_pred}, agg prob: {agg_prob:.3f}")

            # localization for this action segment is done
            if curr_agg_pred != prev_agg_pred and clip_agg_cnt > 0:
                logger.info(f"pred: {prev_agg_pred}, stamps: {(start_time, end_time)}")
                activity_ids.append(prev_agg_pred)
                localization_tuples.append((start_time, end_time))

                # update previous prediction for next batch iter
                prev_agg_pred = curr_agg_pred

                # update start time
                start_time = proposal[2][0]

                # reset all clip aggregation variables
                clip_agg_cnt = 0
                cam_view_clips = {}
            else:
                # update previous prediction for next batch iter
                prev_agg_pred = curr_agg_pred

                # increment count for num clips concatenated consecutively
                clip_agg_cnt += 1

        else:
            # Perform the forward pass.
            preds = model(inputs)

        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1 and cfg.TAL.ENABLE == False:
            preds, labels, video_idx = du.all_gather([preds, labels, video_idx])
        if cfg.NUM_GPUS and cfg.TAL.ENABLE == False:
            preds = preds.cpu()
            labels = labels.cpu()
            video_idx = video_idx.cpu()

        # if proposal is not None:    
        #     # write video_id, pred, max_prob, start_time, end_time to file for post-processing
        #     with open(os.getcwd() + "/post_process/predictions.txt", "a+") as f:
        #         f.writelines(f"{proposal[0][0]} {pred} {max_prob} {proposal[1][0]} {proposal[2][0]}\n")

        test_meter.iter_toc()

        if not cfg.VIS_MASK.ENABLE and cfg.TAL.ENABLE == False:
            # Update and log stats.
            test_meter.update_stats(
                preds.detach(), labels.detach(), video_idx.detach()
            )

        test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    if(cfg.TAL.ENABLE == False):
        # Log epoch stats and print the final testing results.
        if not cfg.DETECTION.ENABLE:
            all_preds = test_meter.video_preds.clone().detach()
            all_labels = test_meter.video_labels
            if cfg.NUM_GPUS:
                all_preds = all_preds.cpu()
                all_labels = all_labels.cpu()
            if writer is not None:
                writer.plot_eval(preds=all_preds, labels=all_labels)

            if cfg.TEST.SAVE_RESULTS_PATH != "":
                save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

                if du.is_root_proc():
                    with pathmgr.open(save_path, "wb") as f:
                        pickle.dump([all_preds, all_labels], f)

                logger.info(
                    "Successfully saved prediction results to {}".format(save_path)
                )

        test_meter.finalize_metrics()
    
    logger.info("Inferencing complete.")

    return test_meter


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    if len(cfg.TEST.NUM_TEMPORAL_CLIPS) == 0:
        cfg.TEST.NUM_TEMPORAL_CLIPS = [cfg.TEST.NUM_ENSEMBLE_VIEWS]

    test_meters = []
    for num_view in cfg.TEST.NUM_TEMPORAL_CLIPS:

        cfg.TEST.NUM_ENSEMBLE_VIEWS = num_view

        # Print config.
        logger.info("Test with config:")
        logger.info(cfg)

        # Build the video model and print model statistics.
        model = build_model(cfg)
        flops, params = 0.0, 0.0
        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            model.eval()
            flops, params = misc.log_model_info(
                model, cfg, use_train_input=False
            )

        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            misc.log_model_info(model, cfg, use_train_input=False)
        if (
            cfg.TASK == "ssl"
            and cfg.MODEL.MODEL_NAME == "ContrastiveModel"
            and cfg.CONTRASTIVE.KNN_ON
        ):
            train_loader = loader.construct_loader(cfg, "train")
            if hasattr(model, "module"):
                model.module.init_knn_labels(train_loader)
            else:
                model.init_knn_labels(train_loader)

        cu.load_test_checkpoint(cfg, model)

        # Create video testing loaders.
        test_loader = loader.construct_loader(cfg, "test")
        logger.info("Testing model for {} iterations".format(len(test_loader)))

        if cfg.DETECTION.ENABLE:
            assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
            test_meter = AVAMeter(len(test_loader), cfg, mode="test")
        else:
            assert (
                test_loader.dataset.num_videos
                % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
                == 0
            )
            # Create meters for multi-view testing.
            test_meter = TestMeter(
                test_loader.dataset.num_videos
                // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
                cfg.MODEL.NUM_CLASSES
                if not cfg.TASK == "ssl"
                else cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM,
                len(test_loader),
                cfg.DATA.MULTI_LABEL,
                cfg.DATA.ENSEMBLE_METHOD,
                cfg.LOG_PERIOD,
            )

        if(cfg.PROMPT.ENABLE == True):
            # create prompt
            prompter = prompters.__dict__[cfg.PROMPT.METHOD](cfg).to("cuda")

            print(f"Using Prompting Method {cfg.PROMPT.METHOD} with Params:")
            if(du.get_rank() == 0):
                for name, param in prompter.named_parameters():
                    if param.requires_grad and '.' not in name:
                        print(name, param.data)

            # optionally resume from a checkpoint
            if cfg.PROMPT.RESUME:
                if os.path.isfile(cfg.PROMPT.RESUME):
                    print("=> loading checkpoint '{}'".format(cfg.PROMPT.RESUME))
                    if cfg.PROMPT.GPU is None:
                        checkpoint = torch.load(cfg.PROMPT.RESUME)
                    else:
                        # Map model to be loaded to specified single GPU.
                        loc = 'cuda:{}'.format(cfg.PROMPT.GPU)
                        checkpoint = torch.load(cfg.PROMPT.RESUME, map_location=loc)
                    cfg.PROMPT.START_EPOCH = checkpoint['epoch']

                    prompter.load_state_dict(checkpoint['state_dict'])
                    print("=> loaded checkpoint '{}' (epoch {})"
                            .format(cfg.PROMPT.RESUME, checkpoint['epoch']))
                else:
                    print("=> no checkpoint found at '{}'".format(cfg.PROMPT.RESUME))
        else:
            prompter = None

        # Set up writer for logging to Tensorboard format.
        if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
            cfg.NUM_GPUS * cfg.NUM_SHARDS
        ):
            writer = tb.TensorboardWriter(cfg)
        else:
            writer = None

        # # Perform multi-view test on the entire dataset.
        test_meter = perform_test(test_loader, model, test_meter, cfg, writer, prompter)
        test_meters.append(test_meter)
        if writer is not None:
            writer.close()

    result_string_views = "_p{:.2f}_f{:.2f}".format(params / 1e6, flops)

    if(cfg.TAL.ENABLE == False):
        for view, test_meter in zip(cfg.TEST.NUM_TEMPORAL_CLIPS, test_meters):
            logger.info(
                "Finalized testing with {} temporal clips and {} spatial crops".format(
                    view, cfg.TEST.NUM_SPATIAL_CROPS
                )
            )
            result_string_views += "_{}a{}" "".format(
                view, test_meter.stats["top1_acc"]
            )

            result_string = (
                "_p{:.2f}_f{:.2f}_{}a{} Top5 Acc: {} MEM: {:.2f} f: {:.4f}"
                "".format(
                    params / 1e6,
                    flops,
                    view,
                    test_meter.stats["top1_acc"],
                    test_meter.stats["top5_acc"],
                    misc.gpu_mem_usage(),
                    flops,
                )
            )

            logger.info("{}".format(result_string))
        logger.info("{}".format(result_string_views))
    else:
        return 
    
    return result_string + " \n " + result_string_views


def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long().to("cuda")
    frames = torch.index_select(frames, 0, index)
    return frames
