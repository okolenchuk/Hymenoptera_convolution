import torch
from scripts.transform import *
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import numpy as np
import torchvision
from torchvision import datasets, models, transforms


use_gpu = torch.cuda.is_available()
device = torch.device("cuda")

path = r'D:\EDUCATION\IT_academy\Test_models\Hymenoptera_convolution\hymenoptera_data\hymenoptera_data'

dataloaders = data_transform(path, batch_size=8)




