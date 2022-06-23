

# StdLib
import os
import json
import math
import numpy as np
import time

# Plt
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
from matplotlib.colors import to_rgb

# Network
import urllib.request
from urllib.error import HTTPError

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


print(f"PyTorch version: {torch.__version__}")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.backends.mps
except AttributeError:
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Is MPS built? Incompatible")
    print(f"Is MPS available? Incompatible")
    print(f"Is CUDA built? {torch.backends.cuda.is_built()}")
    print(f"Is CUDA available? {torch.backends.cuda.is_available()}")
else:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Is MPS built? {torch.backends.mps.is_built()}")
    print(f"Is MPS available? {torch.backends.mps.is_available()}")
    print(f"Is CUDA built? {torch.backends.cuda.is_built()}")
    print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"Using device: {device}")


DATASET_PATH = "../data"
CHECKPOINT_PATH = "../save_models"

os.makedirs(CHECKPOINT_PATH, exist_ok=True)

class GCNLayer(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()

    def forward(self, node_feats, adj_matrix):
        pass

if __name__ == '__main__':

    pass
