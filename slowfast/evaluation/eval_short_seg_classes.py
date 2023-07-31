import numpy as np
from scipy.stats import mode



"""
Given an stdout.log from inference, collects all prediction probabilities and sorts them by class

Only gathers probabilities for scenarios in TAL where both agg and single proposal input result in same pred class

params:
    inf_log: string path of stdout.log file containing pred prob outputs from inference
"""
def gather_inf_class_probs(inf_log:str):
    truepos_probs_by_class = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    truepos_misclassifications_by_class = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    
    falsepos_probs_by_class = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    falsepos_misclassifications_by_class = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

    with open(inf_log, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if 'final prob:' in line:
                prob = float(line.partition('final prob: ')[-1].partition(' tp')[0].strip())
                class_idx = int(lines[i+1].partition('final: (')[-1].partition(',')[0])
                misclasses_str = lines[i+1].partition('segs: ((')[-1].partition('),')[0].split(', ')
                misclasses = [int(x) for x in misclasses_str if x != class_idx]
                
                if 'tp' in line:
                    truepos_probs_by_class[class_idx].append(prob)
                    truepos_misclassifications_by_class[class_idx] += misclasses
                else:
                    falsepos_probs_by_class[class_idx].append(prob)
                    falsepos_misclassifications_by_class[class_idx] += misclasses

    return truepos_probs_by_class, truepos_misclassifications_by_class, falsepos_probs_by_class, falsepos_misclassifications_by_class

"""
Prints mean and median probabilities for each class 
"""
def print_class_stats(truepos_probs_by_class, truepos_misclass_by_class, falsepos_probs_by_class, falsepos_misclass_by_class):
    for class_idx in range(16):
        print(f"Class {class_idx}:")

        truepos_probs = np.array(truepos_probs_by_class[class_idx])
        falsepos_probs = np.array(falsepos_probs_by_class[class_idx])

        if len(truepos_probs) > 0:
            truepos_probs_mode = mode(np.round(truepos_probs,1), keepdims=False)[0]
            truepos_misclass = set(truepos_misclass_by_class[class_idx])

            print(f"\tTrue Positive:\n\tMean = {truepos_probs.mean():.3f}, Median = {np.median(truepos_probs, axis=0):.3f}, Mode = {truepos_probs_mode:.3f}, Max = {np.max(truepos_probs)}, Min = {np.min(truepos_probs)}, Frequency = {len(truepos_probs)}\n\tCommon Misclassifications: {truepos_misclass}\n\t{sorted(truepos_probs, reverse=True)}\n")

        if len(falsepos_probs) > 0:
            falsepos_probs_mode = mode(np.round(falsepos_probs,1), keepdims=False)[0]
            falsepos_misclass = set(falsepos_misclass_by_class[class_idx])

            print(f"\tFalse Positive:\n\tMean = {falsepos_probs.mean():.3f}, Median = {np.median(falsepos_probs, axis=0):.3f}, Mode = {falsepos_probs_mode:.3f}, Max = {np.max(falsepos_probs)}, Min = {np.min(falsepos_probs)}, Frequency = {len(falsepos_probs)}\n\tCommon Misclassifications: {falsepos_misclass}\n\t{sorted(falsepos_probs, reverse=True)}\n\n")


if __name__ == '__main__':  
    log_path = 'inference/submission_files/logs/stdout_re_eval_filter_calibration.log' # Probs consolidated via Gaussian weighted average

    truepos_probs_by_class, truepos_misclass_by_class, falsepos_probs_by_class, falsepos_misclass_by_class = gather_inf_class_probs(log_path)
    print_class_stats(truepos_probs_by_class, truepos_misclass_by_class, falsepos_probs_by_class, falsepos_misclass_by_class)

