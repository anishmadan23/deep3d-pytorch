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
from torchvision import datasets, models, transforms
from tensorboardX import SummaryWriter

from model2 import *
from dataloader import *

########### TensorboardX ###########
LOG_DIR = './logs/'

now = str(datetime.datetime.now())
OUTPUTS_DIR = './outputs/'

if not os.path.exists(LOG_DIR):
	os.makedirs(LOG_DIR)

if not os.path.exists(OUTPUTS_DIR):
	os.makedirs(OUTPUTS_DIR)
OUTPUTS_DIR = OUTPUTS_DIR + now + '/'
if not os.path.exists(OUTPUTS_DIR):
	os.makedirs(OUTPUTS_DIR)
if not os.path.exists(LOG_DIR+now):
	os.makedirs(LOG_DIR+now)

writer = SummaryWriter(LOG_DIR + now)

########### Arguments ###########
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device= torch.device('cpu')
print(device)
max_epoch = 100
# dataroot = '/home/apoorv/Documents/Practice/CV/Project/data_scene_flow/training/'
dataroot = '../KITTI/training/'

batch = 1
save_after = 2
lr = 0.0004
save_file = 'view_syn_weights_l1with_scheduler.pth'
# img_size = (96,320)
img_size = (192,640)

momentum = 0.95
weight_decay = 1.0e-4
resume = False
log = "error_log.txt"

########### Model ###########
model = Deep3d(device=device).to(device)
if(resume):
    model.load_state_dict(torch.load(save_file))
print(model)

########### Dataloader ###########
train_dataset = MyDataset(dataroot, in_transforms = None, size = img_size)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch, shuffle = True)

print(len(train_dataloader))

# Testing dataloader
#for i,data in enumerate(train_dataloader):
#	inn = data[0].float()
#	out = data[1].float()
#	print(inn.shape)
#	print(out.shape)
#sys.exit()

########### Criterion ###########
optimizer = optim.Adam([
        {'params': [param for name, param in model.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * lr},
        {'params': [param for name, param in model.named_parameters() if name[-4:] != 'bias'],
         'lr': lr, 'weight_decay': weight_decay}
], betas=(momentum, 0.999))

lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

########### Begin Training ###########
epoch = 0

while(epoch < max_epoch):
	# lr_scheduler.step()
	for j in ['train']:
		dataloader = train_dataloader
		for i,data in enumerate(dataloader):
			left = data[0].to(device).float()
			right = data[1].to(device).float()

			optimizer.zero_grad()
			output = model(left)
			
			criterion = nn.L1Loss().cuda()
			loss = criterion(output, right)
			loss.backward()
			optimizer.step()
			writer.add_scalar('loss',loss.item())

			print('Epoch={}, iteration={}, input shape={}, output shape={}, loss={}'.format(epoch, i, left.shape, right.shape, loss.item()))

			if i % 50 == 0:
				print(left.shape)
				save_image(left, OUTPUTS_DIR + '{}_{}_scan.png'.format(epoch, i))
				save_image(right, OUTPUTS_DIR + '{}_{}_out.png'.format(epoch, i))			
				save_image(output, OUTPUTS_DIR + '{}_{}_rgb.png'.format(epoch, i))
				torch.save(model.state_dict(), '{}_{}_{}'.format(epoch, i, save_file))

	epoch+=1



