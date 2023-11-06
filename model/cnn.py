import torch.nn as nn

import numpy as np

def affine_relu(in_dim,out_dim):
    return nn.Sequential(
        nn.Linear(in_dim,out_dim)
        ,nn.ReLU()
    )
def conv_batchnorm_relu(in_ch,out_ch,kernel_size,padding=0,stride=1,eps=1e-5):
    return nn.Sequential(
        nn.Conv2d(in_ch,out_ch,kernel_size,padding=padding,stride=stride),
        nn.BatchNorm2d(out_ch,eps)
        ,nn.ReLU()
    )

def conv_relu(in_ch,out_ch,kernel_size,padding=0,stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch,out_ch,kernel_size,padding=padding,stride=stride),
        nn.ReLU()
    )

def calc_conv_dim(inputdim,pad,kernelsize,stride):
    return 1 + int((inputdim + 2 * pad - kernelsize) / stride)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim=128
        num_classes=10
        filters=[32,64,128,128]
        self.batchnorm = nn.BatchNorm2d(3)
        self.conv1=conv_relu(3,filters[0],5,2,stride=2) 
        self.pool1=nn.MaxPool2d(2,2)
        self.conv2=conv_batchnorm_relu(filters[0],filters[1],5,2,stride=2)
        self.pool2=nn.MaxPool2d(2,2)
        self.conv3=conv_batchnorm_relu(filters[1],filters[2],3,1,stride=1)
        self.pool3=nn.MaxPool2d(2,2)
        self.conv4=conv_batchnorm_relu(filters[2],filters[3],3,1,stride=1)


        self.flatten=nn.Flatten()
        self.fc1=affine_relu(2048,512)
        self.fc2=affine_relu(512,128)
        self.fc3=nn.Linear(128,num_classes)

    def forward(self,x):
        x=self.batchnorm(x)
        x=self.conv1(x)
        x=self.pool1(x)
        x=self.conv2(x)
        x=self.pool2(x)
        x=self.conv3(x)
        x=self.pool3(x)
        x=self.conv4(x)
      #  print(x.shape)
        x=self.flatten(x)

       # print(x.shape)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        return x

