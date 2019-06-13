import cv2
import os
import numpy as np 


root_dir = '../2011_09_26_long/2011_09_26_drive_0014_sync/'
left_dir = root_dir+'left/'
right_dir = root_dir+'right/'
genR_dir = root_dir+'genR/'


left_imgs = os.listdir(left_dir)
right_imgs = os.listdir(right_dir)
genR_imgs = os.listdir(genR_dir)

left_imgs.sort(key=lambda x:int(x[:x.find('.')]))
right_imgs.sort(key=lambda x:int(x[:x.find('.')]))
genR_imgs.sort(key=lambda x:int(x[:x.find('.')]))

# left_imgs.sort(key=lambda x:int(x.split('.')[0]))
# # right_imgs.sort(key=lambda x:int(x[:x.find('.')]))
# genR_imgs.sort(key=lambda x:int(x.split('.')[0]))



all_img_types = [left_imgs,right_imgs,genR_imgs]
all_img_dirs = [left_dir,right_dir,genR_dir]

# all_img_types = [left_imgs,genR_imgs]
# all_img_dirs = [left_dir,genR_dir]
for idx,img_type in enumerate(all_img_types):
    for k,img in enumerate(img_type):
        img = cv2.imread(os.path.join(all_img_dirs[idx],img))
        # img = cv2.resize(img,(img.shape[1]*2,img.shape[0]*2), interpolation = cv2.INTER_CUBIC)
        all_img_types[idx][k] = img


new_left_right_dir = '../final_left_right_dir_orig_size/'
if not os.path.exists(new_left_right_dir):
    os.makedirs(new_left_right_dir)


for i in range(len(left_imgs)):
    new_img = np.zeros((left_imgs[i].shape[0]*2,left_imgs[i].shape[1],left_imgs[i].shape[2]))
    new_img[:left_imgs[i].shape[0],:,:] = left_imgs[i]
    new_img[left_imgs[i].shape[0]:,:,:] = right_imgs[i]
    save_name = str(i)+str('.png')
    cv2.imwrite(os.path.join(new_left_right_dir,save_name),new_img)

new_left_genR_dir = '../final_left_genR_dir_orig_size/'
if not os.path.exists(new_left_genR_dir):
    os.makedirs(new_left_genR_dir)


for i in range(len(left_imgs)):
    new_img = np.zeros((left_imgs[i].shape[0]*2,left_imgs[i].shape[1],left_imgs[i].shape[2]))
    new_img[:left_imgs[i].shape[0],:,:] = left_imgs[i]
    new_img[left_imgs[i].shape[0]:,:,:] = genR_imgs[i]
    save_name = str(i)+str('.png')
    cv2.imwrite(os.path.join(new_left_genR_dir,save_name),new_img)

# print(left_imgs[0].shape)
# cv2.imwrite('resized.png',left_imgs[0])
# resized_left_imgs = [cv2.resize(img,(img.shape[1]*2,img.shape[0]*2), interpolation = cv2.INTER_CUBIC) for img in ]



