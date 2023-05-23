import os
import sys
sys.path.insert(1, os.getcwd())
from prepare_data import UCF101_Dataset
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models import ViT_B_32_Weights
from torchinfo import summary
from train_functions import train_model
from models import * 
from torch import nn
from timeit import default_timer as timer 

#################################### SYSTEM PERFORMANCE ######################################

# Always run the start method inside this if-statement
if __name__ == '__main__':      
    print("Setting Multiprocessing Start Method to 'Spawn'")
    # spawn start method instead of fork for CUDA usage
    torch.multiprocessing.set_start_method('spawn')

    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Torch Version: {torch.__version__}")

    torch.set_default_device(device) # set all operations to be done on device by default

    total_free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
    print(f"Total free GPU memory: {round(total_free_gpu_memory * 1e-9, 3)} GB")
    print(f"Total GPU memory: {round(total_gpu_memory * 1e-9, 3)} GB")

    # Get GPU capability score
    GPU_SCORE = torch.cuda.get_device_capability()

    if GPU_SCORE >= (8, 0):
        print(f"[INFO] Using GPU with score: {GPU_SCORE}, enabling TensorFloat32 (TF32) computing (faster on new GPUs)")
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        print(f"[INFO] Using GPU with score: {GPU_SCORE}, TensorFloat32 (TF32) not available, to use it you need a GPU with score >= (8, 0)")
        torch.backends.cuda.matmul.allow_tf32 = False

    ######################################## LOADING THE DATA ############################################

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Load the organized data from the file system into arrays
    data_dir = os.getcwd() + "/image_data"
    data_df = pd.read_csv(data_dir + "/annotation.csv")

    # Remove duplicate Class 1 label named 'Class 01' (Mistake in data labeling)
    data_df['class'].replace("Class 01", "Class 1", inplace=True)

    # turn labels into on

    # create a train-test split
    # however, we must group frames belonging to the same trimmed video clip and assign ids to these groups
    num_video_clips = len(data_df['image'].str.rpartition('_')[0].unique())
    num_frames = len(data_df)
    num_frames_per_clip = int(num_frames/num_video_clips)

    # create a list of group ids for the number of video clips; ids will be duplicated for the number of frames per video
    group_ids = [i for i in range(num_video_clips) for j in range(num_frames_per_clip)]

    data_df['groupID'] = group_ids

    # split data into train and test sets using the group ids
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state = 42)
    split = splitter.split(X=data_df['image'], y=data_df['class'], groups=data_df['groupID'])
    train_indexes, test_indexes = next(split)

    train_df = data_df.iloc[train_indexes]
    test_df = data_df.iloc[test_indexes]

    # reset indexes after split
    train_df.reset_index(inplace=True)
    test_df.reset_index(inplace=True)

    # Create UFC101_Dataset objects (NOTE: change the transform according to input expected by model)
    train_data = UCF101_Dataset(train_df, img_dir=data_dir, transform=ViT_B_32_Weights.IMAGENET1K_V1.transforms())
    test_data = UCF101_Dataset(test_df, img_dir=data_dir, transform=ViT_B_32_Weights.IMAGENET1K_V1.transforms())

    # Setup the batch size hyper param (128 when RAM > 16 GB; 32 when RAM < 16 GB)
    BATCH_SIZE = 128

    # Setting CPU cores available to use for data transfer b/w CPU and GPU
    NUM_WORKERS = os.cpu_count()

    # Turn dataset into iterables (batches)
    train_dataloader = DataLoader(train_data,
                                BATCH_SIZE, 
                                shuffle=True,
                                num_workers=NUM_WORKERS,
                                generator=torch.Generator(device=device))

    test_dataloader = DataLoader(test_data,
                                BATCH_SIZE, 
                                shuffle=False,
                                num_workers=NUM_WORKERS,
                                generator=torch.Generator(device=device))

    num_classes = train_data.num_classes()

    print(f"Number of classes: {num_classes}")

    #################################### RUNNING THE MODEL ######################################

    # Load the model
    #model = Resnt18Rnn(num_classes, dropout_rate=.5, rnn_hidden_size=100, rnn_num_layers=1, batch_size=BATCH_SIZE).to(device)
    model = ViT(num_classes).to(device)
    #model = torch.compile(model_uncompiled) # pytorch 2.0 speed increase

    # Print a summary using torchinfo 
    summary(model=model, 
            input_size=(BATCH_SIZE, 3, 224, 224), # (batch_size, 3 (RGB), 224 (H), 224 (W))
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
    ) 

    # Define loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    torch.autograd.set_detect_anomaly(True)

    start_time = timer()
    results = train_model(model=model,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        epochs=20,
                        device=device)
    end_time = timer()

    total_time = end_time - start_time

    print(f"Total Training Time: {total_time:.2f} seconds")

    # Save results as csv file for later graphing/comparison, etc.
    pd.DataFrame.from_dict(results).to_csv(os.getcwd() + "/Evaluation", header=True, index=False)
