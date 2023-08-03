import os
import numpy as np
from scipy.stats import mode
from ast import literal_eval

"""
Fetches true positive intervals which are annotated in the filter calibration log file
"""
def fetch_tp_fp(log_file):
    tp_intervals = []
    fp_intervals = []

    with open(log_file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if 'final prob:' in line:
                class_idx = lines[i+1].partition('final: (')[-1].partition(',')[0]
                vid_id_str = lines[i+2].partition('vid_id: ')[-1].partition(',')[0].strip()
                timestamp_str = lines[i+2].partition('stamps: ')[-1].strip()

                if 'tp' in line:
                    tp_intervals.append((vid_id_str, timestamp_str, class_idx))
                else:
                    fp_intervals.append((vid_id_str, timestamp_str, class_idx))

    return tp_intervals, fp_intervals


"""
Given an stdout.log from inference, collects all prediction probabilities and sorts them by class

Only gathers probabilities for scenarios in TAL where both agg and single proposal input result in same pred class

params:
    inf_log: string path of stdout.log file containing pred prob outputs from inference
"""
def gather_inf_class_probs(inf_log:str, tp_intervals, fp_intervals, eval_type='[]'):
    truepos_probs_by_class = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    truepos_misclassifications_by_class = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    
    falsepos_probs_by_class = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    falsepos_misclassifications_by_class = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

    truepos_seg_lengths = []
    falsepos_seg_lengths = []

    with open(inf_log, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if 'final prob:' in line:
                prob = float(line.partition('final prob: ')[-1].partition(' tp')[0].strip())
                class_idx = int(lines[i+1].partition('final: (')[-1].partition(',')[0])
                original_pred = int(lines[i+2].partition('pred: ')[-1].partition(',')[0])

                if eval_type == '()':
                    misclasses_str = lines[i+1].partition('segs: ((')[-1].partition('),')[0].split(', ')
                elif eval_type == '[]':
                    misclasses_str = lines[i+1].partition('segs: ([')[-1].partition('],')[0].split(', ')

                misclasses = [int(x) for x in misclasses_str if x != class_idx]

                vid_id_str = lines[i+2].partition('vid_id: ')[-1].partition(',')[0].strip()
                timestamp_str = lines[i+2].partition('stamps: ')[-1].strip()
                interval_str = vid_id_str + ',' + timestamp_str

                interval = literal_eval(interval_str)
                seg_length = interval[1][1] - interval[1][0]
                
                if any(interval_str == (interval_[0] + ',' + interval_[1]) for interval_ in tp_intervals):
                    truepos_probs_by_class[class_idx].append(prob)
                    truepos_misclassifications_by_class[class_idx] += misclasses
                    truepos_seg_lengths.append(seg_length)

                    for _interval in tp_intervals:
                        if interval_str == (_interval[0] + ',' + _interval[1]) and int(_interval[2]) != class_idx:
                            print(f'True Positive, {interval_str}: {_interval[2]} changed to {class_idx}\n')

                    if class_idx != original_pred:
                        print(f'True Positive, {interval_str}: re-eval corrected {original_pred} to {class_idx}')
                
                else:
                    falsepos_probs_by_class[class_idx].append(prob)
                    falsepos_misclassifications_by_class[class_idx] += misclasses
                    falsepos_seg_lengths.append(seg_length)

                    for _interval in fp_intervals:
                        if interval_str == (_interval[0] + ',' + _interval[1]) and int(_interval[2]) != class_idx:
                            print(f'False Positive, {interval_str}: {_interval[2]} changed to {class_idx}\n')

                    if class_idx != original_pred:
                        print(f'False Positive, {interval_str}: re-eval corrected {original_pred} to {class_idx}')

    truepos_seg_lengths = np.array(truepos_seg_lengths)
    falsepos_seg_lengths = np.array(falsepos_seg_lengths)

    print(f"True Positive Segment Length Stats:\n\tMean = {truepos_seg_lengths.mean():.3f}, Median = {np.median(truepos_seg_lengths):.3f}, Mode = {mode(truepos_seg_lengths, keepdims=False)[0]}, Max = {np.max(truepos_seg_lengths)}, Min = {np.min(truepos_seg_lengths)}, Freq = {len(truepos_seg_lengths)}")
    print(f"False Positive Segment Length Stats:\n\tMean = {falsepos_seg_lengths.mean():.3f}, Median = {np.median(falsepos_seg_lengths):.3f}, Mode = {mode(falsepos_seg_lengths, keepdims=False)[0]}, Max = {np.max(falsepos_seg_lengths)}, Min = {np.min(falsepos_seg_lengths)}, Freq = {len(falsepos_seg_lengths)}\n\n")

    return truepos_probs_by_class, truepos_misclassifications_by_class, falsepos_probs_by_class, falsepos_misclassifications_by_class

"""
Prints mean and median probabilities for each class 
"""
def print_class_stats(truepos_probs_by_class, truepos_misclass_by_class, falsepos_probs_by_class, falsepos_misclass_by_class):
    for class_idx in range(16):
        with open(os.getcwd() + '/evaluation/short_seg/eval_short_seg.log', 'a+') as f:
            f.writelines(f"Class {class_idx}:")

            truepos_probs = np.array(truepos_probs_by_class[class_idx])
            falsepos_probs = np.array(falsepos_probs_by_class[class_idx])

        
            if len(truepos_probs) > 0:
                truepos_probs_mode = mode(np.round(truepos_probs,1), keepdims=False)[0]
                truepos_misclass = set(truepos_misclass_by_class[class_idx])

                f.writelines(f"\n\tTrue Positive:\n\tMean = {truepos_probs.mean():.3f}, Median = {np.median(truepos_probs, axis=0):.3f}, Mode = {truepos_probs_mode:.3f}, Max = {np.max(truepos_probs)}, Min = {np.min(truepos_probs)}, Frequency = {len(truepos_probs)}\n\tCommon Misclassifications: {truepos_misclass}\n\t{sorted(truepos_probs, reverse=True)}\n")

            if len(falsepos_probs) > 0:
                falsepos_probs_mode = mode(np.round(falsepos_probs,1), keepdims=False)[0]
                falsepos_misclass = set(falsepos_misclass_by_class[class_idx])

                f.writelines(f"\n\tFalse Positive:\n\tMean = {falsepos_probs.mean():.3f}, Median = {np.median(falsepos_probs, axis=0):.3f}, Mode = {falsepos_probs_mode:.3f}, Max = {np.max(falsepos_probs)}, Min = {np.min(falsepos_probs)}, Frequency = {len(falsepos_probs)}\n\tCommon Misclassifications: {falsepos_misclass}\n\t{sorted(falsepos_probs, reverse=True)}\n\n")


if __name__ == '__main__':  
    anno_log_path = 'inference/submission_files/logs/stdout_re_eval_filter_calibration.log'
    # log_path = 'inference/submission_files/logs/stdout_re_eval_filter_calibration.log' # Probs consolidated via Gaussian weighted average
    log_path = 'inference/submission_files/logs/rework.log'

    tp_intervals, fp_intervals = fetch_tp_fp(anno_log_path)
    truepos_probs_by_class, truepos_misclass_by_class, falsepos_probs_by_class, falsepos_misclass_by_class = gather_inf_class_probs(log_path, tp_intervals, fp_intervals, eval_type='[]')
    print_class_stats(truepos_probs_by_class, truepos_misclass_by_class, falsepos_probs_by_class, falsepos_misclass_by_class)

