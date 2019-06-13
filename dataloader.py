import torch
import torch.utils.data as data
import numpy as np
import torchvision
import os
import scipy.misc as smi

class MyDataset(data.Dataset):
	def __init__(self, root, in_transforms = None, size = (128, 128)):
		self.leftpath = os.path.join(root, 'image_2')
		self.leftimg = os.listdir(self.leftpath)

		self.rightpath = os.path.join(root, 'image_3')
		self.rightimg = os.listdir(self.rightpath)

		self.leftimg.sort()
		self.rightimg.sort()

		# self.leftimg = self.leftimg[::2]
		# self.rightimg = self.rightimg[::2]

		self.size = size

	def __len__(self):
		return len(self.leftimg)

	def __getitem__(self, index):
		leftImage = smi.imread(os.path.join(self.leftpath, self.leftimg[index]))
		# print(leftImage.shape)
		leftImage = smi.imresize(leftImage, self.size, interp='cubic') / 255.0

		rightImage = smi.imread(os.path.join(self.rightpath, self.rightimg[index]))
		rightImage = smi.imresize(rightImage, self.size, interp='cubic') /255.0

		left = torch.from_numpy(leftImage)
		left = left.permute([-1,0,1])
		
		right = torch.from_numpy(rightImage)
		right = right.permute([-1,0,1])

		return left, right

class MyDatasetTest(data.Dataset):
	def __init__(self, root, in_transforms = None, size = (128, 128)):
		self.leftpath = os.path.join(root, 'colored_0/')
		self.leftimg = os.listdir(self.leftpath)

		self.rightpath = os.path.join(root, 'disp_occ/')
		self.rightimg = os.listdir(self.rightpath)

		self.leftimg.sort()
		self.leftimg = self.leftimg[::2]
		self.leftimg.sort()
		self.rightimg.sort()

		# self.rightimg = self.rightimg[::2]

		self.size = size

	def __len__(self):
		return len(self.rightimg)

	def __getitem__(self, index):
		print(self.leftimg[index], self.rightimg[index])
		leftImage = smi.imread(os.path.join(self.leftpath, self.leftimg[index]))
		# print(leftImage.shape)
		leftImage = smi.imresize(leftImage, self.size, interp='cubic') / 255.0

		rightImage = smi.imread(os.path.join(self.rightpath, self.rightimg[index]))
		rightImage = smi.imresize(rightImage, self.size, interp='cubic') /255.0

		left = torch.from_numpy(leftImage)
		left = left.permute([-1,0,1])
		
		right = torch.from_numpy(rightImage)
		# right = right.permute([-1,0,1])

		return left, right

