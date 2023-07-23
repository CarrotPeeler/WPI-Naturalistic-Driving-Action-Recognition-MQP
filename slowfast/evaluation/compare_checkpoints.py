import os
import pandas as pd


"""
THIS FILE RUNS STATISTICS ON INCORRECT PREDICTIONS AND THEIR PROBABILITIES
"""


# Run command:
# cd slowfast
# python3 evaluation/compare_checkpoints.py



"""
Prints incorrect prediction stats for multiple checkpoints for a single model

Helps compare checkpoints to see if overfitting is occuring or if one checkpoint is better than another

args:
    csv_filepath_arr: array of csv filepath strings for each checkpoint being compared 
    model_name: single string of the model being used
    trained_epochs_list: list of integers (number of epochs) for each checkpoint being compared
"""
def print_incorrect_pred_stats(csv_filepath_arr, model_name, trained_epochs_list, save_dir):
    assert len(csv_filepath_arr) == len(trained_epochs_list), "Length of csv_filepath_list does not match length of trained_epochs_list"

    for i in range(len(csv_filepath_arr)):
        stat_type = ["Incorrect", "Correct"]

        checkpoint_df = [pd.read_csv(csv_filepath_arr[i][0], names=["path", "pred", "prob", "target"]),
                         pd.read_csv(csv_filepath_arr[i][1], names=["path", "pred", "prob", "target"])]
        
        inc_pred_conf_rank_df = checkpoint_df[0][["pred", "prob"]].groupby(["pred"]).agg(["mean", "median", "count"]).reset_index()\
            .round(3).sort_values(by=[("prob", "mean"), ("prob", "count")], ascending=[False, False])
        
        stats_str = (f"\n{trained_epochs_list[i]} epochs:\
            \n\n{stat_type[0]} Prediction Stats:\t\t{stat_type[1]} Prediction Stats:\
            \
            \ntotal {stat_type[0].lower()} preds = {len(checkpoint_df[0]['prob'])}\t\ttotal {stat_type[1].lower()} preds = {len(checkpoint_df[1]['prob'])}\
            \
            \n\nmean prob = {checkpoint_df[0]['prob'].mean():.3f}\t\t\tmean prob = {checkpoint_df[1]['prob'].mean():.3f}\
            \
            \nmedian prob = {checkpoint_df[0]['prob'].median():.3f}\t\t\tmedian prob = {checkpoint_df[1]['prob'].median():.3f}\
            \
            \nmax prob = {checkpoint_df[0]['prob'].max():.3f}\t\t\tmax prob = {checkpoint_df[1]['prob'].max():.3f}\
            \
            \nmin prob = {checkpoint_df[0]['prob'].min():.3f}\t\t\tmin prob = {checkpoint_df[1]['prob'].min():.3f}\
            \
            \n\ntotal preds with prob >= 0.9 = {len(checkpoint_df[0][checkpoint_df[0]['prob'] >= 0.9])}\ttotal preds with prob >= 0.9 = {len(checkpoint_df[1][checkpoint_df[1]['prob'] >= 0.9])}\
            \
            \n                       < 0.9 = {len(checkpoint_df[0][checkpoint_df[0]['prob'] < 0.9])}\t                       < 0.9 = {len(checkpoint_df[1][checkpoint_df[1]['prob'] < 0.9])}\
            \
            \n                       < 0.8 = {len(checkpoint_df[0][checkpoint_df[0]['prob'] < 0.8])}\t                       < 0.8 = {len(checkpoint_df[1][checkpoint_df[1]['prob'] < 0.8])}\
            \
            \n\nmean prob by camera angle:\
            \n\nDashboard = {checkpoint_df[0].loc[checkpoint_df[0]['path'].str.partition('_')[0] == 'Dashboard']['prob'].mean():.3f}\t\t\tDashboard = {checkpoint_df[1].loc[checkpoint_df[1]['path'].str.partition('_')[0] == 'Dashboard']['prob'].mean():.3f}\
            \
            \nRearview = {checkpoint_df[0].loc[checkpoint_df[0]['path'].str.partition('_')[0] == 'Rear']['prob'].mean():.3f}\t\t\tRearview = {checkpoint_df[1].loc[checkpoint_df[1]['path'].str.partition('_')[0] == 'Rear']['prob'].mean():.3f}\
            \
            \nRight Side Window = {checkpoint_df[0].loc[checkpoint_df[0]['path'].str.partition('_')[0] == 'Right']['prob'].mean():.3f}\t\tRight Side Window = {checkpoint_df[1].loc[checkpoint_df[1]['path'].str.partition('_')[0] == 'Right']['prob'].mean():.3f}\
            \
            \n\nincorrect pred confidence ranking:\
            \n\n{inc_pred_conf_rank_df}\
            \n")
        
        with open(save_dir + "/" + model_name + f"_{trained_epochs_list[i]}_epochs_checkpoint_stats.txt", "a+") as f:
            f.writelines(stats_str)


if __name__ == '__main__':  

    inc_dir = os.getcwd() + "/evaluation/val_preds/MVITv2_B_32x3_mixup_aug_unprompted/incorrect_preds/"
    cor_dir = os.getcwd() + "/evaluation/val_preds/MVITv2_B_32x3_mixup_aug_unprompted/correct_preds/"
    save_dir = os.getcwd() + "/evaluation/val_preds/MVITv2_B_32x3_mixup_aug_unprompted/checkpoint_stats"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    csv_filepaths = [ #["val_incorrect_pred_probs_mvitv2-b_120_epochs.txt", "val_correct_pred_probs_mvitv2-b_120_epochs.txt"],
                     #["val_incorrect_pred_probs_mvitv2-b_140_epochs.txt", "val_correct_pred_probs_mvitv2-b_140_epochs.txt"],
                     ["val_incorrect_pred_probs_mvitv2-b_160_epochs.txt", "val_correct_pred_probs_mvitv2-b_160_epochs.txt"],
                     ["val_incorrect_pred_probs_mvitv2-b_180_epochs.txt", "val_correct_pred_probs_mvitv2-b_180_epochs.txt"],
                     ["val_incorrect_pred_probs_mvitv2-b_200_epochs.txt", "val_correct_pred_probs_mvitv2-b_200_epochs.txt"]]
    
    for k in range(len(csv_filepaths)):
        csv_filepaths[k][0] = inc_dir + csv_filepaths[k][0]
        csv_filepaths[k][1] = cor_dir + csv_filepaths[k][1]

    print_incorrect_pred_stats(csv_filepath_arr=csv_filepaths,
                               model_name="MViTv2-B",
                               trained_epochs_list=[160, 180, 200],
                               save_dir=save_dir)
    
