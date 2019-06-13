import numpy as np
import cv2 as cv
import cv2
ply_header = '''ply
format ascii 1.0
element vertex 
'''
str2='''
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write(verts, colors):
    verts = verts.reshape(-1, 3)
    vertex=np.zeros((colors.shape[0],6))
    colors = colors.reshape(-1, 3)
    
    for i in range(colors.shape[0]):
        vertex[i][0]=verts[i][0]
        vertex[i][1]=verts[i][1]
        vertex[i][2]=verts[i][2]
        vertex[i][3]=colors[i][0]
        vertex[i][4]=colors[i][1]
        vertex[i][5]=colors[i][2]
    with open('depth.ply', 'wb') as f:
        header=ply_header+str(colors.shape[0])+str2
        f.write((header).encode('utf-8'))
        for i in range(colors.shape[0]):
            string=str(vertex[i][0])+" "+str(vertex[i][1])+" "+str(vertex[i][2])+" "+str(vertex[i][3])+" "+str(vertex[i][4])+" "+str(vertex[i][5])+" "
            f.write((string).encode('utf-8'))
        


def main(filteredImg,imgL,mini):
    disp=filteredImg
    min_disp = 16
    num_disp = 112-min_disp
    (h, w,_) = imgL.shape  
    mask=disp>mini
    Q=np.zeros((4,4))
    Q[0][0]=1
    Q[1][1]=-1
    Q[3][2]=1
    Q[0][3]=-0.5*w
    Q[1][3]=0.5*h
    Q[2][3]=0.8*w
    Q=np.float32(Q)
    points = cv.reprojectImageTo3D(disp, Q)
    colors = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)


    
    out_p = points[mask]
    out_c = colors[mask]

    mask =np.zeros((disp.shape[0],disp.shape[1]),dtype=int)
   
    for i in range(disp.shape[0]):
        for j in range(disp.shape[1]):
            if(disp[i][j]>mini):
                mask[i][j]=1
                # print(mask[i][j])
            else:
                mask[i][j]=0
    
    write(out_p, out_c)
    
 
print('loading images...')
imgL = cv2.imread('l.jpg')
imgR = cv2.imread('R.jpg')
 


window_size = 3                    
 
lm = cv2.StereoSGBM_create(minDisparity=0,numDisparities=16)
 
rm = cv2.ximgproc.createRightMatcher(lm)
 
wlf = cv2.ximgproc.createDisparityWLSFilter(matcher_left=lm)
wlf.setLambda(80000)
wlf.setSigmaColor(1.2)
displ = lm.compute(imgL, imgR) 
dispr = rm.compute(imgR, imgL) 
displ = np.int16(displ)
dispr = np.int16(dispr)
final = wlf.filter(displ, imgL, None, dispr)
final = np.uint8(final)
main(final,imgL,final.min())
cv.destroyAllWindows()
