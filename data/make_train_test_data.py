import os
import numpy as np 
import shutil
import glob
import random

random.seed(0)


# trainval data folders
train_val_data_dirs = ['2011_09_26_drive_0011_sync','2011_09_26_drive_0022_sync','2011_09_26_drive_0059_sync','2011_09_26_drive_0084_sync'
						,'2011_09_26_drive_0093_sync','2011_09_26_drive_0095_sync','2011_09_26_drive_0096_sync']
# test data folders
test_data_dirs = ['2011_09_26_drive_0019_sync','2011_09_26_drive_0091_sync']

train_dir = './train/'
val_dir = './val/'
test_dir = './test/'

if not os.path.exists(train_dir):
	os.makedirs(train_dir)
if not os.path.exists(val_dir):
	os.makedirs(val_dir)
if not os.path.exists(test_dir):
	os.makedirs(test_dir)


all_left_img_paths = []
all_right_img_paths = []
left_suffix = 'image_02/data'
right_suffix = 'image_03/data'
for idx, trainval_dir in enumerate(train_val_data_dirs):
	left_imgs = glob.glob(os.path.join(trainval_dir,left_suffix,'*.png'))
	right_imgs = glob.glob(os.path.join(trainval_dir,right_suffix,'*.png'))

	left_imgs.sort(key=lambda x:int(x.split('/')[-1][:x.split('/')[-1].find('.png')]))
	right_imgs.sort(key=lambda x:int(x.split('/')[-1][:x.split('/')[-1].find('.png')]))

	all_left_img_paths.extend(left_imgs)
	all_right_img_paths.extend(right_imgs)

	### sanity check to see if number of left and right images match
	# print(len(left_imgs))
	# print(len(right_imgs))

combined_list = list(zip(all_left_img_paths,all_right_img_paths))
random.shuffle(combined_list)
all_left_img_paths,all_right_img_paths = zip(*combined_list)

### sanity check to see if random hasn't spoiled left-right pairs
# print(all_left_img_paths[:5])
# print(all_right_img_paths[:5])

train_idx = int(0.8*len(all_left_img_paths))     # 80-20 split
train_left_imgs_path = all_left_img_paths[:train_idx]
train_right_imgs_path = all_right_img_paths[:train_idx]

val_left_imgs_path = all_left_img_paths[train_idx:]
val_right_imgs_path = all_right_img_paths[train_idx:]

### train data
for idx in range(len(train_left_imgs_path)):
	base_img_name = '00000000'      # eight zeros
	img_name = (len(base_img_name) - len(str(idx)))*'0' + str(idx)+'.png'

	new_left_train_path = os.path.join(train_dir,'left')
	new_right_train_path = os.path.join(train_dir,'right')

	if not os.path.exists(new_left_train_path):
		os.makedirs(new_left_train_path)
	if not os.path.exists(new_right_train_path):
		os.makedirs(new_right_train_path)

	shutil.copy(train_left_imgs_path[idx],os.path.join(new_left_train_path,img_name))
	shutil.copy(train_right_imgs_path[idx],os.path.join(new_right_train_path,img_name))


### val data
for idx in range(len(val_left_imgs_path)):
	base_img_name = '00000000'      # eight zeros
	img_name = (len(base_img_name) - len(str(idx)))*'0' + str(idx)+'.png'

	new_left_val_path = os.path.join(val_dir,'left')
	new_right_val_path = os.path.join(val_dir,'right')

	if not os.path.exists(new_left_val_path):
		os.makedirs(new_left_val_path)
	if not os.path.exists(new_right_val_path):
		os.makedirs(new_right_val_path)

	shutil.copy(val_left_imgs_path[idx],os.path.join(new_left_val_path,img_name))
	shutil.copy(val_right_imgs_path[idx],os.path.join(new_right_val_path,img_name))


test_left_imgs_path = []
test_right_imgs_path = []
### test data
for idx, test_dir_ in enumerate(test_data_dirs):
	left_imgs = glob.glob(os.path.join(test_dir_,left_suffix,'*.png'))
	right_imgs = glob.glob(os.path.join(test_dir_,right_suffix,'*.png'))

	left_imgs.sort(key=lambda x:int(x.split('/')[-1][:x.split('/')[-1].find('.png')]))
	right_imgs.sort(key=lambda x:int(x.split('/')[-1][:x.split('/')[-1].find('.png')]))

	test_left_imgs_path.extend(left_imgs)
	test_right_imgs_path.extend(right_imgs)

# print(len(test_left_imgs_path))

for idx in range(len(test_left_imgs_path)):
	base_img_name = '00000000'      # eight zeros
	img_name = (len(base_img_name) - len(str(idx)))*'0' + str(idx)+'.png'

	new_left_test_path = os.path.join(test_dir,'left')
	new_right_test_path = os.path.join(test_dir,'right')

	if not os.path.exists(new_left_test_path):
		os.makedirs(new_left_test_path)
	if not os.path.exists(new_right_test_path):
		os.makedirs(new_right_test_path)
		
	shutil.copy(test_left_imgs_path[idx],os.path.join(new_left_test_path,img_name))
	shutil.copy(test_right_imgs_path[idx],os.path.join(new_right_test_path,img_name))

