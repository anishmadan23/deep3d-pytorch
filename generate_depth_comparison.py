import numpy as np
import cv2
import sys
import os 
import math

from sklearn.preprocessing import normalize
from cv2 import ximgproc, StereoSGBM_create
 
def generate_depth(imgL, imgR):

    window_size = 7
    
    lm = StereoSGBM_create(minDisparity=0,numDisparities=16,blockSize=5,P1=8 * 3 * window_size ** 2,P2=32 * 3 * window_size ** 2,disp12MaxDiff=1,)
    
    rm = ximgproc.createRightMatcher(lm)
    
    wlf = ximgproc.createDisparityWLSFilter(matcher_left=lm)
    wlf.setLambda(80000)
    wlf.setSigmaColor(1.2)
    displ = lm.compute(imgL, imgR) 
    dispr = rm.compute(imgR, imgL) 
    final = wlf.filter(displ, imgL, None, dispr)
    final = final - 2*np.min(final)
    final = final*255/np.max(final)
    final = np.uint8(final)
    return final

INPUT_DIR = './results/2019-04-29 12:29:15.506219/'
OUTPUT_DIR = './depth/'
list_img = os.listdir(INPUT_DIR)
list_img.sort()

mae = []
rmse = []
for i in range(0,len(list_img),3):
    print(list_img[i], list_img[i+1], list_img[i+2])
    imgL = cv2.imread(INPUT_DIR+list_img[i+2])
    imgR = cv2.imread(INPUT_DIR+list_img[i+1])
    depth_map_out = generate_depth(imgL, imgR)
    # depth_map_out = depth_map_out / np.max(depth_map_out)
    # depth_map_out_s = depth_map_out * 255
    # depth_map_out[0:30,:] = 0
    
    imgRG = cv2.imread(INPUT_DIR+list_img[i])
    # imgRG = cv2.cvtColor(imgRG, cv2.COLOR_BGR2GRAY)
    # imgRG = imgRG / np.max(imgRG)
    # imgRGS = imgRG * 255
    depth_map_ground = generate_depth(imgL, imgRG)

    if(np.max(depth_map_ground)>1):
        depth_map_ground = depth_map_ground / 255.0 * 65.0 + 1.0
    if(np.max(depth_map_out)>1):
        depth_map_out = depth_map_out / 255.0 * 65.0 + 1.0

    cv2.imwrite(OUTPUT_DIR+'{}_g.png'.format(i), depth_map_ground.astype(np.uint8))
    cv2.imwrite(OUTPUT_DIR+'{}_o.png'.format(i), depth_map_out.astype(np.uint8))

    diff = np.abs(depth_map_out - depth_map_ground)
    mae.append(np.sum(diff)/(depth_map_out.shape[0]*depth_map_out.shape[1])/depth_map_ground)
    rmse.append(np.sum(diff**2)/(depth_map_out.shape[0]*depth_map_out.shape[1]))

    #sys.exit()
rmse = np.array(rmse)
rmse = np.mean(rmse**(0.5))
mae = np.array(mae)
mae = np.mean(mae)

print('RMSE = {}, MAE = {}'.format(rmse, mae))