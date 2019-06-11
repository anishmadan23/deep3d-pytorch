# deep3d-pytorch
Estimating a right-view from a monocular image(to make a stereo pair of images) which respects the geometry of the scene is 
an important problem in computer vision. This repository aims to achieve this by implementing 
[Deep3D](https://arxiv.org/abs/1604.03650) ([see original repo](https://github.com/piiswrong/deep3d)) using PyTorch 
to generate right view of images. This generated pair of images can then be used to estimate depth in images, 
convert 2D video to 3D, etc.
