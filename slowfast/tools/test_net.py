#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
import sys

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
from inference.TAL import *

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, models, test_meter, cfg, writer=None, prompter=None):
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
        models (model): the pretrained video models to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model = models[0]
    model_2 = models[1]

    model.eval()
    model_2.eval()

    test_meter.iter_tic()

    if cfg.PROMPT.ENABLE == True and prompter is not None:
        prompter.eval()

    # delete existing predictions.txt if exists
    pred_output = os.getcwd() + "/post_process/predictions.txt"
    if os.path.exists(pred_output):
        print("DELETING predictions.txt")
        os.remove(pred_output)


    if cfg.TAL.ENABLE == True:
        start_time = None
        end_time = None
        video_id = None
        prev_agg_pred = None
        prev_consol_code = None

        # stores aggregated clips (batches) for each cam view type
        cam_view_clips = {}

        # num clips aggregated for current temporal action interval
        clip_agg_cnt = 0

        # num of aggregated frames to keep from last iteration
        frame_agg_threshold = cfg.TAL.CLIP_AGG_THRESHOLD - cfg.DATA.NUM_FRAMES 
        

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
                
            # AGGREGATION STEP: aggregate clip frames together in order to increase temporal resolution for classification
            if cur_iter == 0: 
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

                # only set start_time here if 1st iter; otherwise, its set when prev_agg_pred changes
                
                start_time = proposal[1][0]

                end_time = proposal[2][0]

                video_id = proposal[0][0]

            else:
                # this proposal may contain a new action pred, so use start time of this proposal as end time to prev action pred
                end_time = proposal[1][0]

                cviews = set()
                        
                # check that there's no duplicate cam views and video ids + start/end times are matching
                # Concat clips to their respective batch (based on cam view type) 
                for j in range(cfg.TEST.BATCH_SIZE):
                    cview = test_loader.dataset._path_to_videos[video_idx[j]].rpartition('/')[-1].partition('_user')[0]
                    cviews.add(cview)

                    # fix the aggregation clip size to be constant past the threshold 
                    start_idx = cam_view_clips[cview].shape[0] - frame_agg_threshold if clip_agg_cnt*cfg.DATA.NUM_FRAMES > frame_agg_threshold else 0 

                    cam_view_clips[cview] = torch.cat([cam_view_clips[cview][start_idx:], inputs[0][j].permute(1,0,2,3)], dim=0)

                    assert cam_view_clips[cview].shape[1] == 3 \
                        and cam_view_clips[cview].shape[2] == cfg.DATA.TEST_CROP_SIZE \
                        and cam_view_clips[cview].shape[3] == cfg.DATA.TEST_CROP_SIZE, f"Shape mismatch, got {cam_view_clips[cview].shape}"

                assert len(cviews) == 3, f"Cam view mismatch for next batch {cur_iter}"
                assert len(set(proposal[0])) == len(set(proposal[1])) == len(set(proposal[2])) == 1, f"Proposal mismatch for next batch {cur_iter}"

            # logger.info(f"CUR ITER: {cur_iter}")
            
            # PREDICTION STEP: make predictions for each camera angle using identical proposal length and space
            preds, probs, preds_2, probs_2 = predict_cam_views(cfg, model, model_2, cam_view_clips, frame_agg_threshold, logger, resample=False)

            labels = labels.cpu()
            video_idx = video_idx.cpu()

            # CONSOLIDATION STEP: Consolidate predictions among the three camera angles into 1 final pred
            curr_agg_pred, curr_consol_code = consolidate_preds(preds, probs, cfg.TAL.FILTERING_THRESHOLD, logger)
            

            # CORRECTION STEP: correct prediction if necessary
            fix_start_time = False # signal if start time needs adjustment

            # detect new action w/ common pred but prob is low => resample current temporal interval and consolidate again
            if curr_agg_pred != prev_agg_pred and curr_consol_code == -1:
                logger.info("Perform Resampling")
                preds, probs, _, _ = predict_cam_views(cfg, model, model_2, cam_view_clips, frame_agg_threshold, logger, resample=True)
                
                curr_agg_pred, curr_consol_code = consolidate_preds(preds, probs, cfg.TAL.FILTERING_THRESHOLD, logger)

                # if final consol code is 0, resampling worked and new action begins 1s later instead of at start of new interval
                if curr_consol_code == 0:
                    proposal_temporal_resolution = float(proposal[2][0]) - float(proposal[1][0])
                    end_time = str(float(end_time) + (proposal_temporal_resolution/2.0))

                    # re-evaluated precision of localization for new detected action, adjust start time accordingly
                    fix_start_time = True

            if len(preds_2) == 3: agg_pred_2, curr_consol_code_2 = consolidate_preds(preds_2, probs_2, cfg.TAL.FILTERING_THRESHOLD, logger)

            # invalidate results if curr_agg_pred differs from agg_pred_2 OR one of the preds is invalid
            if curr_consol_code in [-1,-2] or curr_consol_code_2 in [-2] or curr_agg_pred != agg_pred_2:
                curr_consol_code = -1


            # ASSESSMENT STEP: check if temporal localization for the current action is done
            """ 
            only finish localization of an action when one of following occurs:
                - video_id changes
                - curr and prev preds differ
                - there are issues predicting over the current temporal interval => toss results and reset aggregation of clips 
            """
            vid_id_changed = (video_id != proposal[0][0])
            action_changed = (curr_agg_pred != prev_agg_pred and clip_agg_cnt > 0)
            curr_pred_is_bad = curr_consol_code in [-1, -2] 

            if vid_id_changed or curr_pred_is_bad or action_changed:
                logger.info(f"prev code: {prev_consol_code}, cur code: {curr_consol_code}")

                # if vid or action changed and prev pred is valid, record the temporal interval of the prev action
                if prev_consol_code == 0 or clip_agg_cnt > 1:
                    start_time = int(float(start_time)//1)
                    end_time = int(float(end_time)//1)

                    with open(cfg.TAL.OUTPUT_FILE_PATH.rpartition('.')[0] + "_unmerged.txt", "a+") as f:
                        f.writelines(f"{video_id} {prev_agg_pred} {start_time} {end_time}\n")
                    
                    logger.info(f"vid_id: {video_id}, pred: {prev_agg_pred}, stamps: {(start_time, end_time)}")

                # set start time of newly detected action 
                start_time = proposal[1][0]
                
                # adjust start time if precision of localization for new detected action has been updated
                if fix_start_time:
                    proposal_temporal_resolution = float(proposal[2][0]) - float(proposal[1][0])
                    start_time = str(float(start_time) + (proposal_temporal_resolution/2.0))

                # update video id
                video_id = proposal[0][0]

                # reset all clip aggregation variables
                clip_agg_cnt = 1
                del cam_view_clips
                cam_view_clips = {}
                
                # re-add this iteration's frames from the clip
                for b in range(cfg.TEST.BATCH_SIZE):
                    cview = test_loader.dataset._path_to_videos[video_idx[b]].rpartition('/')[-1].partition('_user')[0]
                    cam_view_clips[cview] = inputs[0][b].permute(1,0,2,3)

            else: 
                # continue localization for current action pred

                # increment count for num clips concatenated consecutively
                clip_agg_cnt += 1

            # update previous prediction for next batch iter as well as consolidation code
            prev_consol_code = curr_consol_code
            prev_agg_pred = curr_agg_pred

        ######################## T A L  E N D ########################

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

        test_meter.iter_toc()

        if not cfg.VIS_MASK.ENABLE and cfg.TAL.ENABLE == False:
            # Update and log stats.
            test_meter.update_stats(
                preds.detach(), labels.detach(), video_idx.detach()
            )

        test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    if cfg.TAL.ENABLE == False:
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
    else:
        # merge unmerged segments
        post_process_merge(cfg.TAL.OUTPUT_FILE_PATH.rpartition('.')[0] + "_unmerged.txt", cfg.TAL.OUTPUT_FILE_PATH)
    
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
        model = build_model(cfg, 0)

        if cfg.TAL.ENABLE == True and cfg.TAL.USE_2_GPUS == True:
            model_2 = build_model(cfg, 1)
        else:
            model_2 = model

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

        if cfg.TAL.ENABLE == True and cfg.TAL.USE_2_GPUS == True:
            cu.load_test_checkpoint(cfg, model_2)

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
        test_meter = perform_test(test_loader, [model, model_2], test_meter, cfg, writer, prompter)
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