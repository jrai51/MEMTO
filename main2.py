import os
import argparse
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set to GPU 0 or as per your configuration

print('------------ 1st CUDA CHECK -------------')

print("CUDA Available inside main.py:", torch.cuda.is_available())
print("Device Count inside main.py:", torch.cuda.device_count())

