import os
import torch
import torchvision

import numpy as np
import skimage.io as io
import torch.utils.data as data

from skimage.transform import resize

class MyDataset(data.Dataset):
	def __init__(self, root, in_transforms = None, orig_size = (384, 1280),small_size=(96,320)):
		self.leftpath = os.path.join(root, 'left')
		self.leftimg = os.listdir(self.leftpath)

		self.rightpath = os.path.join(root, 'right')
		self.rightimg = os.listdir(self.rightpath)

		self.leftimg.sort()
		self.rightimg.sort()

		self.orig_size = orig_size
		self.small_size= small_size

	def __len__(self):
		return len(self.leftimg)

	def __getitem__(self, index):
		leftImage = io.imread(os.path.join(self.leftpath, self.leftimg[index]))
		# print(leftImage.shape)
		leftImage_orig = resize(leftImage, self.orig_size) / 255.0

		leftImage_small = resize(leftImage, self.small_size) / 255.0

		rightImage_orig = io.imread(os.path.join(self.rightpath, self.rightimg[index]))
		rightImage_orig = resize(rightImage_orig, self.orig_size) /255.0

		left_orig = torch.from_numpy(leftImage_orig)
		left_orig = left_orig.permute([-1,0,1])

		left_small = torch.from_numpy(leftImage_small)
		left_small = left_small.permute([-1,0,1])
		
		right_orig = torch.from_numpy(rightImage_orig)
		right_orig = right_orig.permute([-1,0,1])

		return left_orig, left_small, right_orig


# data_obj = MyDataset('./data/train/')
# train_dataloader = data.DataLoader(data_obj, batch_size = 4, shuffle = True)



