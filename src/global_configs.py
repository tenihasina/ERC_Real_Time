import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROGRAM"] = "multimodal_driver.py"

DEVICE = torch.device("cuda:0")

# KAMUH
ACOUSTIC_DIM = 300
VISUAL_DIM = 300
TEXT_DIM = 768

XLNET_INJECTION_INDEX = 1
