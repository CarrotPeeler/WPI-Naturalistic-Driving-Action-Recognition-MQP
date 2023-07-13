import os
import math
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt

# run command
# cd slowfast
# python3 evaluation/graph_val_acc.py


"""
Graphs the top1 validation accuracy scores to compare different training sessions' results

params:
    json_stats: list of string paths to json stat files used for graphing
    epochs: num epochs for graphing over x-axis
    legend_labels: list of legend labels for each line (order should match json_stats list)
"""
def graph_top1_val_acc(json_stat_paths:list, epochs:int, eval_freq:int, legend_labels:list, figtitle:str):
    x = list(range(1, epochs+1, eval_freq))
    y = []

    for json_stat_path in json_stat_paths:
        y_vals = []
        with open(json_stat_path) as f:
            for line in f:
                if "val_epoch" in line:
                    y_vals.append(float(line.rpartition('top1_acc":')[-1].partition(',')[0]))
        y.append(y_vals)

    y_lim_min = math.floor(min([y[i][0] for i in range(len(y))]))
    y_lim_max = math.ceil(max([max(y[i]) for i in range(len(y))]))

    fig = plt.figure(figsize=(17, 6))

    color_cycle = cycle('bgrcmk')

    for idx, y_val in enumerate(y):
        ax = fig.add_subplot(1, len(y), idx+1)
        ax.plot(x, y_val, next(color_cycle))
        # ax.scatter(x, y_val, s=5)

        # # create trend line
        # z = np.polyfit(x, y_val, 50)
        # p = np.poly1d(z)

        # #add trendline to plot
        # plt.plot(x, p(x), next(color_cycle))

        ax.set_title(legend_labels[idx])
        ax.xaxis.set_ticks(np.arange(0, epochs+1, 50))
        ax.set_ylim(y_lim_min, y_lim_max)

        plt.xlabel('Time (epochs)')
        plt.ylabel('Accuracy (%)')

    fig.suptitle(figtitle)

    fig.subplots_adjust(top=0.85)

    plt.savefig(os.getcwd() + "/evaluation/graphs/val_acc_graphs/val_acc_compare.png") # save the figure to file
    
    print("Figure saved.")


if __name__ == '__main__':

    num_epochs = 200
    eval_freq = 2
    dir_path = '/home/vislab-001/Jared/Naturalistic-Driving-Action-Recognition-MQP/slowfast/evaluation/mvitv2-b32x3/' 
    
    json_paths = [
        dir_path + 'normal_data/Json_stats_mvitv2-b_round1_unprompted_base-lr_2e-4_end-lr_2e-6_30epoch-warmup-lr_2e-6.log',
        dir_path + 'prompted_multicam_padding/selective_updating/best_results/Json_stats_mvitv2-b_round1_selective_updating_multicam_padding_30_lr_0.1_wd_1e-4_base-lr_2e-4_end-lr_2e-6_30epoch-warmup-lr_2e-6.log',
        dir_path + 'prompted_multicam_padding/selective_updating/best_results/json_stats_mvitv2-b_round1_selective_updating_multicam_padding_30_lr_0.02_wd_1e-4_base-lr_2e-4_end-lr_2e-6_30epoch-warmup-lr_2e-6.log'
    ]

    legend_labels = [
        'unprompted training',
        'selective updating, multicam padding\nprompt LR = 0.1',
        'selective updating, multicam padding\nprompt LR = 0.02'
    ]

    figtitle = "Top1 Validation Accuracy Over Time for MViTv2-B (Base LR = 2e-4, start and end LR = 2e-6, 30 epoch warmup)"

    graph_top1_val_acc(json_paths, num_epochs, eval_freq, legend_labels, figtitle)