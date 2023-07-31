import numpy as np
from scipy.stats import mode



"""
Given an stdout.log from inference, collects all prediction probabilities and sorts them by class

Only gathers probabilities for scenarios in TAL where both agg and single proposal input result in same pred class

params:
    inf_log: string path of stdout.log file containing pred prob outputs from inference

returns:
    two lists (for aggregated and single proposal preds) w/ lists of probs, sorted by class index
"""
def gather_inf_class_probs(inf_log:str):
    agg_probs_by_class = [[],[],[],[],
                          [],[],[],[],
                          [],[],[],[],
                          [],[],[],[]]
    prop_probs_by_class =[[],[],[],[],
                          [],[],[],[],
                          [],[],[],[],
                          [],[],[],[]]

    with open(inf_log, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i+1 < len(lines) and 'AGG pred:' in line and 'AGG pred:' in lines[i+1]:
                class_idx_l1 = int(line.partition('AGG pred: ')[-1].partition(',')[0])
                class_idx_l2 = int(lines[i+1].partition('AGG pred: ')[-1].partition(',')[0])

                if class_idx_l1 == class_idx_l2:
                    agg_prob = float(line.partition(', prob: ')[-1].partition(',')[0])
                    agg_probs_by_class[class_idx_l1].append(agg_prob)

                    prop_prob = float(lines[i+1].partition(', prob: ')[-1].partition(',')[0])
                    prop_probs_by_class[class_idx_l2].append(prop_prob)

    return agg_probs_by_class, prop_probs_by_class


"""
Prints mean and median probabilities for each class 

params:
    agg_probs_by_class: list w/ lists of probs, sorted by class index
    prop_probs_by_class: list w/ lists of probs, sorted by class index
"""
def print_class_stats(agg_probs_by_class, prop_probs_by_class):
    for i in range(16):
        assert len(agg_probs_by_class[i]) == len(prop_probs_by_class[i]), f'Shape mismatch, class {i}'
    
    for class_idx in range(len(agg_probs_by_class)):
        agg_class_probs = np.array(agg_probs_by_class[class_idx])
        prop_class_probs = np.array(prop_probs_by_class[class_idx])
        agg_mode = mode(agg_class_probs, keepdims=False)[0]
        prop_mode = mode(prop_class_probs, keepdims=False)[0]

        print(f"Class {class_idx}:")
        print(f"AGG:\n\tMean = {agg_class_probs.mean():.3f}, Median = {np.median(agg_class_probs, axis=0):.3f}, Max = {np.max(agg_class_probs)}, Min = {np.min(agg_class_probs)}, Mode = {agg_mode}, Frequency = {len(agg_class_probs)}")
        print(f"PROP:\n\tMean = {prop_class_probs.mean():.3f}, Median = {np.median(prop_class_probs, axis=0):.3f}, Max = {np.max(prop_class_probs)}, Min = {np.min(prop_class_probs)} Mode = {prop_mode}, Frequency = {len(prop_class_probs)}")


if __name__ == '__main__':  
    log_path = 'inference/submission_files/logs/stdout_low_filter_only_neg_2.log'

    agg_probs_by_class, prop_probs_by_class = gather_inf_class_probs(log_path)
    print_class_stats(agg_probs_by_class, prop_probs_by_class)

