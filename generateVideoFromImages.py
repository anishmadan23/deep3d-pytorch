import cv2
import os
import numpy as np 

my_dir = '../india_gate_left_genR_dir_orig_size/'

all_imgs = os.listdir(my_dir)
all_imgs.sort(key=lambda x:int(x[:x.find('.')]))

imgs_for_video = []
for img in all_imgs:
    img = cv2.imread(os.path.join(my_dir,img))
    height, width, layers = img.shape
    size = (width,height)
    imgs_for_video.append(img)


fourcc = cv2.VideoWriter_fourcc(*'MP4V') # Be sure to use lower case
out = cv2.VideoWriter('india_gate_long_orig_3d_vid.avi', fourcc, 10.0, (width, height))
# out = cv2.VideoWriter('long_orig_3d_vid.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(imgs_for_video)):
    print(i)
    out.write(imgs_for_video[i])
out.release()

