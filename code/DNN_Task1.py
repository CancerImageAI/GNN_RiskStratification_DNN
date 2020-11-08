"""
Created on Wed Jul  3 13:24:41 2019

@author: PC
"""

from torch.nn import Module, Sequential 
from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool1d
from torch.nn import ReLU, Sigmoid
from torch import nn
import torch
import torch.nn.functional as F

class Conv3D_Block(Module):
        
    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, dilation=1, residual=None):
        
        super(Conv3D_Block, self).__init__()

        self.conv1 = Sequential(
                        Conv3d(inp_feat, out_feat, kernel_size=kernel, 
                                    stride=stride, padding=padding, dilation=dilation, bias=True),
                        BatchNorm3d(out_feat),
                        ReLU())

        self.conv2 = Sequential(
                        Conv3d(out_feat, out_feat, kernel_size=kernel, 
                                    stride=stride, padding=padding, dilation=dilation, bias=True),
                        BatchNorm3d(out_feat))
        
        
        
        self.residual = residual
        self.relu = ReLU()
        if self.residual is not None:
            self.residual_upsampler = Sequential(
                    Conv3d(inp_feat, out_feat, kernel_size=1, stride=stride, bias=False),
                    BatchNorm3d(out_feat))

    def forward(self, x):
        
        res = x

        if not self.residual:
            return self.conv2(self.conv1(x))
        else:
            return self.relu(self.conv2(self.conv1(x)) + self.residual_upsampler(res))

class Conv3D_Block0(Module):
        
    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual=None):
        
        super(Conv3D_Block0, self).__init__()

        self.conv = Sequential(
                        Conv3d(inp_feat, out_feat, kernel_size=kernel, 
                                    stride=stride, padding=padding, bias=True),
                        BatchNorm3d(out_feat),
                        ReLU())

        
        self.residual = residual
        self.relu = ReLU()
        if self.residual is not None:
            self.residual_upsampler = Sequential(
                    Conv3d(inp_feat, out_feat, kernel_size=1, stride=stride, bias=False),
                    BatchNorm3d(out_feat))

    def forward(self, x):
        
        res = x
        print(self.residual_upsampler(res).size())
        if not self.residual:
            return self.conv1(x)
        else:
            return self.relu(self.conv(x) + self.residual_upsampler(res))
        
def Maxpool3D_Block():
    
    pool = MaxPool3d(kernel_size=2, stride=2, padding=0)
    
    return pool       
        

    
class ClassifyNet(Module):
    def __init__(self, num_feat=[16,32,64,128,256], residual='conv'):
        super(ClassifyNet, self).__init__()
        
        #Encoder process
        self.conv1_0 = Conv3D_Block(1, num_feat[0], residual=residual)
        self.conv1_1 = Conv3D_Block(1, num_feat[0], padding=2, dilation=2, residual=residual)
        self.pool1 = Maxpool3D_Block()
        self.conv2_0 = Conv3D_Block(num_feat[0]*2, num_feat[1], residual=residual)
        self.conv2_1 = Conv3D_Block(num_feat[0]*2, num_feat[1], padding=2, dilation=2, residual=residual)
        self.pool2 = Maxpool3D_Block()
        self.conv3_0 = Conv3D_Block(num_feat[1]*2, num_feat[2], residual=residual)
        self.conv3_1 = Conv3D_Block(num_feat[1]*2, num_feat[2], padding=2, dilation=2, residual=residual)
        self.pool3 = Maxpool3D_Block()
        self.conv4_0 = Conv3D_Block(num_feat[2]*2, num_feat[3], residual=residual)
        # self.conv4_1 = Conv3D_Block(num_feat[3], num_feat[3], padding=2, dilation=2,residual=residual)
        self.pool4 = Maxpool3D_Block()
        self.conv5 = Conv3D_Block(num_feat[3], num_feat[4], residual=residual)
#        self.conv6 = Conv3D_Block(num_feat[4], num_feat[4], residual=residual)
#        self.drop = nn.Dropout(p = 0.5)

        self.fc1 = nn.Linear(num_feat[4]*2*2*2,2)
#        self.fc2 = nn.Linear(512,2)
#        self.fc3 = nn.Linear(256,2)
        self.Relu = nn.ReLU()


    def forward(self, x):
        
        down_1_0 = self.conv1_0(x)
        down_1_1 = self.conv1_1(x)
        down_1 = torch.cat([down_1_0,down_1_1],dim=1)
        pool_1 = self.pool1(down_1)
        down_2_0 = self.conv2_0(pool_1)
        down_2_1 = self.conv2_1(pool_1)
        down_2 = torch.cat([down_2_0,down_2_1],dim=1)
        pool_2 = self.pool2(down_2)
        down_3_0 = self.conv3_0(pool_2)
        down_3_1 = self.conv3_1(pool_2)
        down_3 = torch.cat([down_3_0,down_3_1],dim=1)
        pool_3 = self.pool3(down_3)
        down_4_0 = self.conv4_0(pool_3)
#        down_4_1 = self.conv4_1(down_4_0)
        pool_4 = self.pool4(down_4_0)
        down_5 = self.conv5(pool_4)  


#        down_6 = self.conv6(down_5)  
        view1 = down_5.view(down_5.size(0),-1)
#        view1 = self.drop(view1)
#        fc1 = self.Relu(self.fc1(view1))
#        fc1 = self.drop(fc1)
#        out = F.avg_pool3d(down_5, 4)
#        out = out.view(out.size(0), -1)
        out = self.fc1(self.Relu(view1))
        
        return out
