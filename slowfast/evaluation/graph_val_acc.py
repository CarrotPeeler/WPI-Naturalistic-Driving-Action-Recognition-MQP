import os
import math
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# run command
# cd slowfast
# python3 evaluation/graph_val_acc.py


"""
Graphs the top1 validation accuracy scores to compare different training sessions' results

params:
    json_stats: list of string paths to json stat files used for graphing
    epochs: num epochs for graphing over x-axis
    legend_labels: list of legend labels for each line (order should match json_stats list)
    min_epoch_window: min epoch for which accuracy scores will be displayed
    sliding_window: window size for moving average
    tracked_accs: creates a horizontal line for particular accuracy values (to help better compare results)
"""
def graph_top1_val_acc(json_stat_paths:list, epochs:int, legend_labels:list, figtitle:str, min_epoch_window=1, sliding_window=10, tracked_accs=[80]):
    x_validation = []
    x_train = list(range(1, epochs+1))
    y_validation = []
    y_train = []

    for json_stat_path in json_stat_paths:
        y_vals = []
        y_trains = []
        with open(json_stat_path) as f:
            for line in f:
                if "val_epoch" in line:
                    y_vals.append(float(line.rpartition('top1_acc":')[-1].partition(',')[0]))
                elif "train_epoch" in line:
                    y_trains.append(float(line.rpartition('top1_acc":')[-1].partition(',')[0]))
        y_validation.append(y_vals)
        y_train.append(y_trains)

        eval_freq = int(epochs/len(y_vals))
        x_validation.append(list(range(min_epoch_window, epochs+1, eval_freq)))

    fig = plt.figure(figsize=(17, 6))

    color_cycle = cycle('bgrcmk')

    min_epoch_window_idx = math.ceil(len(y_validation[0])*min_epoch_window/epochs)

    y_lim_min = math.floor(min([min(y_validation[i][min_epoch_window_idx:]) for i in range(len(y_validation))]))
    y_lim_max = math.ceil(max([max(y_validation[i][min_epoch_window_idx:]) for i in range(len(y_validation))]))

    for idx in range(len(y_validation)):
        
        y_validation_moving_avg = []

        for i in range(min_epoch_window_idx-1, len(y_validation[0])):
            avg = sum(y_validation[idx][i - sliding_window: i])/sliding_window
            y_validation_moving_avg.append(avg)

        ax = fig.add_subplot(1, len(y_validation), idx+1)
        # ax.plot(x_validation, y_validation[idx], next(color_cycle))
        # ax.plot(x_train, y_train[idx], next(color_cycle))
        c = next(color_cycle)

        # ax.scatter(x_validation, y_validation[idx], s=5, c=c)
        ax.plot(x_validation[idx], y_validation[idx][min_epoch_window_idx-1:], c, alpha=0.2, linestyle="dashdot")
        ax.plot(x_validation[idx], y_validation_moving_avg, c)

        for acc in tracked_accs:
            ax.plot(x_validation[idx], [acc]*len(x_validation[idx]), 'k', alpha=0.2, linestyle='dashed')

        # # create trend line
        # z = np.polyfit(x_validation, y_validation[idx], 2)
        # p = np.poly1d(z)

        # coeff_of_determination = r2_score(y_validation[idx], p(x_validation))

        # #add trendline to plot
        # ax.text(0.7, 0.3, f'R$^2$: {coeff_of_determination:.3}', fontsize = 15, transform=ax.transAxes)
        # ax.plot(x_validation, p(x_validation), c)

        ax.set_title(legend_labels[idx])
        # ax.xaxis.set_ticks(np.arange(0, epochs+1, 50))
        ax.set_ylim(y_lim_min, y_lim_max)

        ax.legend(['raw accuracy scores', f'moving average scores\n(window size = {sliding_window})'], loc='lower right')

        plt.xlabel('Time (epochs)')
        plt.ylabel('Accuracy (%)')

    fig.suptitle(figtitle)

    fig.subplots_adjust(top=0.80)

    plt.savefig(os.getcwd() + "/evaluation/graphs/val_acc_graphs/val_acc_compare.png") # save the figure to file
    
    print("Figure saved.")


if __name__ == '__main__':

    num_epochs = 200
    dir_path = '/home/vislab-001/Jared/Naturalistic-Driving-Action-Recognition-MQP/slowfast/evaluation/mvitv2-b32x3/' 
    
    json_paths = [
        # dir_path + 'prompted_multicam_padding/json_stats_mvitv2-b_round1_multicam_padding_30_lr_0.02_wd_1e-4_base-lr_9e-5_end-lr_1e-6_warmup-lr_1e-6.log',
        # dir_path + 'prompted_multicam_padding/selective_updating/json_stats_mvitv2-b_round1_selective_updating_multicam_padding_30_lr_0.02_wd_1e-4_base-lr_9e-5_end-lr_1e-6_warmup-lr_1e-6.log'
        # dir_path + 'normal_data/Json_stats_mvitv2-b_round1_unprompted_base-lr_2e-4_end-lr_2e-6_30epoch-warmup-lr_2e-6.log',
        # dir_path + 'prompted_multicam_padding/v2/Json_stats_mvitv2-b_round1_selective_updating_multicam_paddingv2_30_lr_40_wd_1e-4_base-lr_2e-4_end-lr_2e-6_30epoch-warmup-lr_2e-6.log',
        # dir_path + 'prompted_multicam_padding/v2/Json_stats_mvitv2-b_round1_selective_updating_multicam_paddingv2_30_lr_40_wd_1e-4_base-lr_2e-4_end-lr_2e-6_30epoch-warmup-lr_2e-6_trial2.log',
        # dir_path + 'prompted_multicam_padding/v2/Json_stats_mvitv2-b_round1_selective_updating_multicam_paddingv2_30_lr_0.1_wd_1e-4_base-lr_2e-4_end-lr_2e-6_30epoch-warmup-lr_2e-6.log',
        # dir_path + 'prompted_padding/Json_stats_mvitv2-b_round1_padding_30_lr_0.05_wd_1e-4_base-lr_2e-4_end-lr_2e-6_30epoch-warmup-lr_2e-6.log',
        # dir_path + 'prompted_multicam_padding/Json_stats_mvitv2-b_round1_multicam_padding_30_lr_0.1_wd_1e-4_base-lr_2e-4_end-lr_2e-6_30epoch-warmup-lr_2e-6.log',
        # dir_path + 'prompted_multicam_padding/selective_updating/best_results/Json_stats_mvitv2-b_round1_selective_updating_multicam_padding_30_lr_0.1_wd_1e-4_base-lr_2e-4_end-lr_2e-6_30epoch-warmup-lr_2e-6.log',
        # dir_path + 'prompted_multicam_padding/selective_updating/best_results/Json_stats_mvitv2-b_round1_selective_updating_multicam_padding_30_lr_0.1_wd_1e-4_base-lr_2e-4_end-lr_2e-6_30epoch-warmup-lr_2e-6_pretrained_prompt_200epochs.log',
        # dir_path + 'prompted_multicam_padding/selective_updating/Json_stats_mvitv2-b_round1_selective_updating_multicam_padding_30_lr_0.1_wd_1e-4_base-lr_2e-4_end-lr_2e-6_30epoch-warmup-lr_2e-6_pretrained_prompt_400epochs.log',
        # dir_path + 'prompted_multicam_padding/selective_updating/best_results/json_stats_mvitv2-b_round1_selective_updating_multicam_padding_30_lr_0.02_wd_1e-4_base-lr_2e-4_end-lr_2e-6_30epoch-warmup-lr_2e-6.log'
        dir_path + 'MVITv2_B_32x3_mixup_aug_unprompted/json_stats_MVITv2_B_32x3_mixup_aug_unprompted.log',
        dir_path + 'MVITv2_B_32x3_mixup_aug_unprompted/json_stats_MVITv2_B_32x3_mixup_aug_unprompted_pretrained200.log',
    ]

    legend_labels = [
        # 'multicam padding\nprompt LR = 0.02',
        # 'selective updating, multicam padding\nprompt LR = 0.02'
        # 'unprompted training',
        # 'selective updating, multicam padding v2\nprompt LR = 40',
        # 'selective updating, multicam padding v2\nprompt LR = 40, trial 2',
        # 'selective updating, multicam padding v2\nprompt LR = 0.1',
        # 'padding\nprompt LR = 0.05',
        # 'multicam padding\nprompt LR = 0.1',
        # 'selective updating, multicam padding\nprompt LR = 0.1',
        # 'selective updating, multicam padding\npre-trained prompt (200 epochs)',
        # 'selective updating, multicam padding\npre-trained prompt (400 epochs)',
        # 'selective updating, multicam padding\nprompt LR = 0.02'
        'unprompted mixup/augment training, 200 epochs',
        'unprompted mixup/augment training, 400 epochs',
    ]

    figtitle = "Top1 Validation Accuracy Over Time for MViTv2-B\n(Base LR = 5e-4, start and end LR = 1e-6, 35 epoch warmup)"
    # figtitle = "Top1 Validation Accuracy Over Time for MViTv2-B\n(Base LR = 9e-5, start and end LR = 1e-6, 15 epoch warmup)"

    graph_top1_val_acc(json_paths, num_epochs, legend_labels, figtitle, min_epoch_window=100, sliding_window=10, tracked_accs=[83, 84, 85])