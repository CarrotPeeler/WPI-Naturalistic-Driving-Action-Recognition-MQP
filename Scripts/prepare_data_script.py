import os
import sys
sys.path.insert(1, os.getcwd())
from prepare_data import *
import torch

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

######################################### PREPARE DATA ########################################

    user_data_dirs = os.listdir("/home/vislab-001/Jared/SET-A1")

    for user_data in user_data_dirs:
        

    train_1_dir = os.getcwd() + "/train_1"

    # truncate each train video into frames (truncate_size = num of frames per video)
    videoToFrames(train, "/home/vislab-001/Jared/ucf101", train_1_dir, truncate_size=16)

    # Create annotation file (lists all frame filenames and their labels)
    create_annotation(train_1_dir, train_1_dir, "train_1_annotation.csv")