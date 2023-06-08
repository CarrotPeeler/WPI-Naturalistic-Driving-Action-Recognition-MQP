import os
import torch
import ast
from glob import glob

# Always run the start method inside this if-statement
if __name__ == '__main__':  
    
    list = torch.tensor([.1, .2222, 1.3, 4.5, .0009]).tolist()
    a = str(list)
    b = ast.literal_eval(a)
    print(b)
