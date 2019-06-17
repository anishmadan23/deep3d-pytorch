import sys
import math
import time
import datetime
import os
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F

from torchvision.utils import make_grid, save_image
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from tensorboardX import SummaryWriter

from model2 import *
from dataloader import *

RES_DIR = './results/'
now = str(datetime.datetime.now())

if not os.path.exists(RES_DIR):
    os.makedirs(RES_DIR)
if not os.path.exists(RES_DIR+now):
    os.makedirs(RES_DIR+now)
RES_DIR = RES_DIR + now + '/'

device = torch.device('cuda')
print(device)
dataroot = './data/test/'
weight_file = './72_296_view_syn_weights_l1with_scheduler.pth'
batch = 1
img_size = (96, 320)

model = Deep3d(device=device).to(device)
model.load_state_dict(torch.load(weight_file))

test_dataset = MyDataset(dataroot, in_transforms = None)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch, shuffle = False)

print(len(test_dataloader))

model.eval()
for i, data in enumerate(
    test_dataloader):
    with torch.no_grad():
        left_orig = data[0].to(device).float()
        left = data[1].to(device).float()
        right = data[2].to(device).float()
        
        output = model(left_orig,left)

        save_image(left_orig, RES_DIR + '{}_left.png'.format(i))
        save_image(right, RES_DIR + '{}_right.png'.format(i))            
        save_image(output, RES_DIR + '{}_.genR.png'.format(i))
        print(output.shape, left.shape, right.shape)
        # sys.exit()
