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
dataroot = '/home/apoorv/Documents/Practice/CV/Project/data_scene_flow/testing'
weight_file = './99_20_view_syn_weights_l1with_scheduler.pth'
batch = 1
img_size = (96, 320)

model = Deep3d(device=device).to(device)
model.load_state_dict(torch.load(weight_file))

test_dataset = MyDataset(dataroot, in_transforms = None, size = img_size)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch, shuffle = True)

print(len(test_dataloader))

model.eval()
for i, data in enumerate(test_dataloader):
    with torch.no_grad():
        if(i>100):
            break
        left = data[0].to(device).float()
        right = data[1].to(device).float()

        output = model(left)

        save_image(left, RES_DIR + '_{}_scan.png'.format(i))
        save_image(right, RES_DIR + '_{}_out.png'.format(i))            
        save_image(output, RES_DIR + '_{}_rgb.png'.format(i))
        print(output.shape, left.shape, right.shape)
        # sys.exit()
