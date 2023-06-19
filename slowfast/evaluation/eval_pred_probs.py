import os
import pandas as pd

# Run command:
# cd slowfast



if __name__ == '__main__':  
    
    df_120 = pd.read_csv(os.getcwd() + "/evaluation/val_incorrect_pred_probs_mvitv2-b_120_epochs.txt", names=["path", "pred", "prob", "target"])
    df_240 = pd.read_csv(os.getcwd() + "/evaluation/val_incorrect_pred_probs_mvitv2-b_240_epochs.txt", names=["path", "pred", "prob", "target"])

    print(f"\nMViTv2-B stats among incorrect predictions:\n \
          \n120 epochs:\ttotal incorrect preds = {len(df_120['prob'])}\nmean = {df_120['prob'].mean():.3f}\nmedian = {df_120['prob'].median():.3f}\nmax = {df_120['prob'].max():.3f}\nmin = {df_120['prob'].min():.3f}\n \
          \n240 epochs:\ttotal incorrect preds = {len(df_240['prob'])}\nmean = {df_240['prob'].mean():.3f}\nmedian = {df_240['prob'].median():.3f}\nmax = {df_240['prob'].max():.3f}\nmin = {df_240['prob'].min():.3f}\n")