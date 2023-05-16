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

    image_dir = os.getcwd() + "/image_data"

    # truncate each train video into frames (truncate_size = num of frames per video)
    videosToFrames(video_dir="/home/vislab-001/Jared/SET-A1", frame_dir=image_dir, video_extension=".MP4", truncate_size=16)