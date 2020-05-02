# Deep3D-pytorch
## Team Members: Anish Madan, Apoorv Khattar, Yash Tomar

## About the porject
Estimating a right-view from a monocular image(to make a stereo pair of images) which respects the geometry of the scene is 
an important problem in computer vision. This repository aims to achieve this by implementing 
[Deep3D](https://arxiv.org/abs/1604.03650) ([see original repo](https://github.com/piiswrong/deep3d)) using PyTorch 
to generate right view of images. This generated pair of images can then be used to estimate depth in images, 
convert 2D video to 3D, etc.

## Dataset
We used the KITTI Stereo 2015 Dataset. The dataset consists of 200 training scenes and 200 test scenes, which include 4 color images per scene, in a lossless png format. This means that we have 400 left and right image pairs for training.

## Model Weights
The pretrained model can be downloaded from [here](https://drive.google.com/drive/folders/1txjqUjCcEvEkVS8QvNn1icrP34eh-crJ?usp=sharing) 

## Results
The following are some results of our approach (from left to right- input left , ground truth right stereo image pair, generated stereo image pair, depth map generated using OpenCV for image pairs):

![s1](https://github.com/anishmadan23/deep3d-pytorch/blob/master/o1.png)

![s2](https://github.com/anishmadan23/deep3d-pytorch/blob/master/o2.png)

###### This work was done as part of our project for CSE344: Computer Vision course at IIIT Delhi.
