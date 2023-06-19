import os
import pandas as pd


"""
THIS FILE RUNS STATISTICS ON INCORRECT PREDICTIONS AND THEIR PROBABILITIES
"""


# Run command:
# cd slowfast



"""
Prints incorrect prediction stats for multiple checkpoints for a single model

Helps compare checkpoints to see if overfitting is occuring or if one checkpoint is better than another

args:
    csv_filepath_list: list of csv filepath strings for each checkpoint being compared
    model_name: single string of the model being used
    trained_epochs_list: list of integers (number of epochs) for each checkpoint being compared
"""
def print_incorrect_pred_stats(csv_filepath_list, model_name, trained_epochs_list):
    assert len(csv_filepath_list) == len(trained_epochs_list), "Length of csv_filepath_list does not match length of trained_epochs_list"
    
    print(f"\n{model_name} stats for incorrect predictions:\n")

    for i in range(len(csv_filepath_list)):
        checkpoint_df = pd.read_csv(csv_filepath_list[i], names=["path", "pred", "prob", "target"])
        
        print(f"\n{trained_epochs_list[i]} epochs:\ttotal incorrect preds = {len(checkpoint_df['prob'])}\
              \n\nmean prob = {checkpoint_df['prob'].mean():.3f}\
              \nmedian prob = {checkpoint_df['prob'].median():.3f}\
              \nmax prob = {checkpoint_df['prob'].max():.3f}\
              \nmin prob = {checkpoint_df['prob'].min():.3f}\
              \n\ntotal preds with prob >= 0.9 = {len(checkpoint_df[checkpoint_df['prob'] >= 0.9])}\
              \ntotal preds with prob < 0.9 = {len(checkpoint_df[checkpoint_df['prob'] < 0.9])}\
              \ntotal preds with prob < 0.8 = {len(checkpoint_df[checkpoint_df['prob'] < 0.8])}\
              \n")


if __name__ == '__main__':  

    csv_filepaths = [os.getcwd() + "/evaluation/val_incorrect_pred_probs_mvitv2-b_120_epochs.txt",
                     os.getcwd() + "/evaluation/val_incorrect_pred_probs_mvitv2-b_240_epochs.txt"]

    print_incorrect_pred_stats(csv_filepath_list=csv_filepaths,
                               model_name="MViTv2-B",
                               trained_epochs_list=[120, 240])
    
