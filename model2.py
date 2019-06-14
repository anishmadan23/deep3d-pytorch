import torch 
import torch.nn as nn
import numpy as np 
import torchvision
import torch.nn.functional as F

cfg = [64, 128, 256, 512, 512]

class Deep3d(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, device=torch.device('cpu')):
        super(Deep3d, self).__init__()
        self.device = device
        vgg16 = torchvision.models.vgg16_bn(pretrained=True)
        modules = []
        layer = []
        for l in vgg16.features:
            if isinstance(l, nn.MaxPool2d):
                layer.append(l)
                modules.append(layer)
                layer = []
            else:
                layer.append(l)

        scale = 1
        deconv = []
        layer = []
        for m in range(len(modules)):
            layer.append(nn.Conv2d(cfg[m], cfg[m], kernel_size=3, stride=1, padding=True))
            layer.append(nn.ReLU(inplace=True))

            layer.append(nn.Conv2d(cfg[m], cfg[m], kernel_size=3, stride=1, padding=True))
            layer.append(nn.ReLU(inplace=True))

            if(m==0):
                layer.append(nn.ConvTranspose2d(cfg[m], 65, kernel_size=1, stride=1, padding=(0,0)))

            else:
                scale *=2
                layer.append(nn.ConvTranspose2d(cfg[m], 65, kernel_size=scale*2, stride=scale, padding=(scale//2, scale//2)))


            deconv.append(layer)           # add blocks of layers to deconv part of the network
            layer = []

        self.module_1 = nn.Sequential(*modules[0])
        self.module_2 = nn.Sequential(*modules[1])
        self.module_3 = nn.Sequential(*modules[2])
        self.module_4 = nn.Sequential(*modules[3])
        self.module_5 = nn.Sequential(*modules[4])

        self.deconv_1 = nn.Sequential(*deconv[0])
        self.deconv_2 = nn.Sequential(*deconv[1])
        self.deconv_3 = nn.Sequential(*deconv[2])
        self.deconv_4 = nn.Sequential(*deconv[3])
        self.deconv_5 = nn.Sequential(*deconv[4])

        self.linear_module = nn.Sequential(*[nn.Linear(15360,4096),          # hyperparam choice
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(p=0.5),
                                            nn.Linear(4096,1950)])          # 1950=65(disparity range)*10*3(10*3 is feature map size)

        # scale*=2
        self.deconv_6 = nn.Sequential(*[nn.ConvTranspose2d(65,65,kernel_size=scale*2,stride=scale,padding=(scale//2,scale//2))])

        self.upconv_final = nn.Sequential(*[nn.ConvTranspose2d(65,65,kernel_size=(4,4),stride=2,padding=(1,1)),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(65,65,kernel_size=(3,3),stride=1,padding=(1,1)),
                                            nn.Softmax(dim=1)])

        for block in [self.deconv_1,self.deconv_2,self.deconv_3,self.deconv_4,self.deconv_5,self.deconv_6,self.linear_module,self.upconv_final]:
        # for block in [self.deconv_1,self.deconv_2,self.deconv_3,self.deconv_4,self.deconv_5,self.deconv_6,self.upconv_final]:
            for m in block:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    
    def forward(self, orig_x, x):

        x_copy = orig_x
        pred = []
        out_1 = self.module_1(x)
        out_2 = self.module_2(out_1)
        out_3 = self.module_3(out_2)
        out_4 = self.module_4(out_3)
        out_5 = self.module_5(out_4)
        print(out_5.shape)

        out_5_flatten = out_5.view(x_copy.shape[0],-1)
        out_6 = self.linear_module(out_5_flatten)

        p1 = self.deconv_1(out_1)

        p2 = self.deconv_2(out_2)
        p3 = self.deconv_3(out_3)
        p4 = self.deconv_4(out_4)
        p5 = self.deconv_5(out_5)
        # print(p5.shape)
        p6 = self.deconv_6(out_6.view(x_copy.shape[0],65,3,10))

        pred.append(p1)
        pred.append(p2)
        pred.append(p3)
        pred.append(p4)
        pred.append(p5)
        pred.append(p6)
        
        out = torch.zeros(pred[0].shape).to(self.device)
        for p in pred:
            # print('p',p.shape)
            # print('out',out.shape)

            out = torch.add(out, p)

        out = self.upconv_final(out)               # to be elt wise multiplied with shifted left views

        print('out shape',out.shape)
        out = F.interpolate(out,scale_factor=4,mode='bilinear')
        print('new_out shape',out.shape)

        new_right_image = torch.zeros(x_copy.size()).to(self.device)
        stacked_shifted_view = None
        stacked_out = None
        for depth_map_idx in range(-33,32):
            shifted_input_view = torch.zeros(x_copy.size()).to(self.device)
           
            if depth_map_idx<0:
                shifted_input_view[:,:,:,:depth_map_idx] = x_copy[:,:,:,-depth_map_idx:]
            elif depth_map_idx==0:
                shifted_input_view = x_copy
            else:
                # print(depth_map_idx)
                shifted_input_view[:,:,:,depth_map_idx:] = x_copy[:,:,:,:-depth_map_idx]
            # print(shifted_input_view.shape)

            if stacked_shifted_view is None:
                stacked_shifted_view = shifted_input_view.unsqueeze(1)
            else:
                stacked_shifted_view = torch.cat((stacked_shifted_view,shifted_input_view.unsqueeze(1)),dim=1)

            if stacked_out is None:
                stacked_out = out[:,depth_map_idx+33:depth_map_idx+34,:,:].unsqueeze(1)
            else:
                stacked_out = torch.cat((stacked_out,out[:,depth_map_idx+33:depth_map_idx+34,:,:].unsqueeze(1)),dim=1)
        # print("SO",stacked_out.shape)
        # print(stacked_shifted_view.shape)
        #     new_right_image += torch.mul(shifted_input_view,out[:,depth_map_idx+33:depth_map_idx+34,:,:])
        # return new_right_image
        softmaxed_stacked_shifted_view = stacked_shifted_view
        # softmaxed_stacked_shifted_view = F.softmax(stacked_shifted_view,dim=1)
        # print(softmaxed_stacked_shifted_view.shape)

        mult_soft_shift_out = torch.mul(stacked_out,softmaxed_stacked_shifted_view)
        # print('mult',mult_soft_shift_out.shape)

        final_rt_image = torch.sum(mult_soft_shift_out,dim=1)
        print('final_rt_image.shape',final_rt_image.shape)

        return final_rt_image





if(__name__=='__main__'):
    vgg16 = torchvision.models.vgg16(pretrained=True)
    print(vgg16)
    model = Deep3d().to(torch.device('cpu'))
    out = model(torch.randn(10,3,384,1280),torch.randn(10,3,96,320))
    # model(torch.randn(1,3,320,96))
 