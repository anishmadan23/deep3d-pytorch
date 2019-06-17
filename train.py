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

train_dataroot = './data/train/'
val_dataroot = './data/val/'

batch = 2
save_after = 2
lr = 0.0004
save_file = 'view_syn_weights_l1with_scheduler.pth'
img_size = (96,320)


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
train_dataset = MyDataset(train_dataroot, in_transforms = None)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch, shuffle = True)

val_dataset = MyDataset(val_dataroot, in_transforms=None)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = batch, shuffle=False)

dataloaders = {'train': train_dataloader, 'val': val_dataloader}
dataset_sizes = {'train': len(train_dataset),'val':len(val_dataset)}
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
best_loss = 100000
while(epoch < max_epoch):
	since = time.time()
	criterion = nn.L1Loss().cuda()

	print('Epoch {}/{}'.format(epoch, max_epoch - 1))
	print('-' * 10)

	for phase in ['train','val']:
		if phase=='train':
			lr_scheduler.step()
			model.train()
		else:
			model.eval()

		running_loss = 0.0
		
		for iteration,data in enumerate(dataloaders[phase]):
			left_orig = data[0].to(device).float()
			left = data[1].to(device).float()
			right = data[2].to(device).float()

			optimizer.zero_grad()

			with torch.set_grad_enabled(phase == 'train'):
				output = model(left_orig,left)
				loss = criterion(output, right)

				if phase=='train':
					loss.backward()
					optimizer.step()

					writer.add_scalar('loss',loss.item())

			print('Epoch {}, Iteration: {}, Loss: {}'.format(epoch,iteration,loss.item()))
			running_loss += loss.item() * left.size(0)

			if iteration % 200 == 0 and phase=='val':
				# print(left.shape)
				save_image(left_orig, OUTPUTS_DIR + '{}_{}_scan.png'.format(epoch, iteration))
				save_image(right, OUTPUTS_DIR + '{}_{}_out.png'.format(epoch, iteration))			
				save_image(output, OUTPUTS_DIR + '{}_{}_rgb.png'.format(epoch, iteration))
		

		epoch_loss = running_loss / dataset_sizes[phase]

		print('{} Loss: {:.4f}'.format(phase, epoch_loss))

		if phase == 'val' and running_loss<best_loss:
			best_loss = running_loss
			torch.save(model.state_dict(), '{}_{}_{}'.format(epoch, iteration, save_file))

	time_elapsed = time.time() - since
	print('Epoch Time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

	epoch+=1



