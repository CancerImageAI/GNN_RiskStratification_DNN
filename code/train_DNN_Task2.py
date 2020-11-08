# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:24:56 2020

@author: Administrator
"""



import os
import time
import numpy as np
from DNN_Task2 import *
import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from importlib import import_module
import pandas as pd
from scipy.ndimage.interpolation import rotate
from skimage import exposure
import xlrd
from DataAugmentation import *
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score,auc

class GGODataGenerator(Dataset):
    def __init__(self, phase='train', crop_size=32, move=3):
        assert(phase == 'train' or phase == 'val' or phase == 'test')
        data = xlrd.open_workbook('GGN_all_64.xls')
        table = data.sheets()[0]
        
        Malig_ind_all = np.where(np.array(table.col_values(2))=='malignant')[0]    
        IA_ind = np.where(np.array(table.col_values(3))=='IA')[0]
        self.IA = np.array(table.col_values(0))[IA_ind]
        self.IA_Path = np.array(table.col_values(4))[IA_ind]
        self.List_Num = np.array(table.col_values(0))[Malig_ind_all]
        self.Class = np.array(table.col_values(3))[Malig_ind_all]
        self.img_path = np.array(table.col_values(4))[Malig_ind_all]
        self.transform = Transform(crop_size, move)
        self.phase = phase
                
    def __getitem__(self,idx):  
        if self.phase =='train':
            if len(self.List_Num)<=idx<(len(self.List_Num)+len(self.IA)):
                IA = True
                idx = idx%len(self.List_Num)

            elif (len(self.List_Num)+len(self.IA))<=idx<(len(self.List_Num)+len(self.IA)*2):
                IA = True
                idx = idx%(len(self.List_Num)+len(self.IA))

            else:
                IA = False


        if self.phase == 'train':
            if IA: 
                dcm_File = self.IA[idx]
                roi_path = self.IA_Path[idx]+'/'+dcm_File+'.npy'
                ROI = np.load(roi_path)
                ROI = self.transform(ROI)
                Class = 1          
            else:
                dcm_File = self.List_Num[idx]
                roi_path = self.img_path[idx]+'/'+dcm_File+'.npy'
                ROI = self.transform(np.load(roi_path))
                if self.Class[idx] == 'IA':
                    Class =1
                else:
                    Class =0
                    
        return np.array(ROI)[np.newaxis,...], np.array(Class)
        
    def __len__(self):
        if self.phase == 'train':
            return len(self.List_Num)+len(self.IA)*2
        else:
            return len(self.List_Num)


class VD_DataGenerator(Dataset):
    def __init__(self, phase='test', crop_size=32, move=None):
        data = xlrd.open_workbook('../VD/GGN_all_64.xls')
        table = data.sheets()[0]
        Malig_ind_all = np.where(np.array(table.col_values(2))=='malignant')[0]    
        self.List_Num = np.array(table.col_values(0))[Malig_ind_all]
        self.Class = np.array(table.col_values(3))[Malig_ind_all]
        self.img_path = '../VD/GGO_Crop_64'
        self.transform = Transform(crop_size, move)
        self.phase = phase
        
    def __getitem__(self,idx):
        if self.phase == 'test':
            dcm_File = self.List_Num[idx]
            roi_path = self.img_path+'/'+dcm_File+'.npy'
            ROI = self.transform(np.load(roi_path))
            if self.Class[idx] == 'IA':
                Class =1
            else:
                Class =0
        return np.array(ROI)[np.newaxis,...], np.array(Class)
        
    def __len__(self):
        return len(self.List_Num)
    
def get_lr(epoch, lr):
    if epoch <= epochs * 0.5:
        lr = lr
    elif epoch <= epochs * 0.8:
        lr = 0.1 * lr
    else:
        lr = 0.01 * lr
    return lr


def mixup_data(x, y, alpha=1.0, use_cuda=True):
 
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
 
    mixed_x = lam * x + (1 - lam) * x[index,:] # 自己和打乱的自己进行叠加
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam
 
def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(data_loader, data_loader_VD, net, loss, epoch, optimizer, lr_scheduler, save_freq, save_dir):
    starttime = time.time()
    net.train()
    lr_scheduler.step()
    train_loss = 0
    correct = 0
    total = 0
    for i, (data,Class) in enumerate(data_loader):  
#        Mixup Augmentation
        inputs, targets = data.cuda(), Class.cuda()
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets.long(), 0.5, True)
        optimizer.zero_grad() 
        inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
        output = net(inputs)
        loss_func = mixup_criterion(targets_a, targets_b, lam)

        loss_output = loss_func(loss, output)
        loss_output.backward()  
        optimizer.step()
        train_loss += loss_output.data.cpu().numpy()
        _, predicted = torch.max(output.data,1)  
        total += len(targets)        
        correct += lam * predicted.eq(targets_a.data).cpu().sum().numpy() + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().numpy()
    metrics = np.asarray(train_loss/(i+1), np.float32)
    acc = np.asarray(100.*correct/total, np.float32)
    print('Epoch %03d' % (epoch))
    print('ResNet loss %2.4f' % (np.mean(metrics)))
    print('ResNet Accuracy %2.4f' % (np.mean(acc)))
    

    test_loss = []
    Label_VD = []
    ResNet_prob_VD = []
    for i, (data,test_Class) in enumerate(data_loader_VD):
        Label_VD.append(test_Class)
        data = Variable(data.cuda())
        target = Variable(test_Class.cuda())
        net.eval()
        with torch.no_grad():
            output = F.softmax(net(data), dim = 1)
            loss_output = loss(output,target.long())
        ResNet_result = output.data.cpu().numpy()
        ResNet_prob_VD.append(ResNet_result[0][1])
        test_loss.append(loss_output)

    test_loss = np.asarray(test_loss, np.float32)
    acc = accuracy_score(np.array(Label_VD),(np.array(ResNet_prob_VD)>0.5).astype(int))*100
    fpr,tpr,threshold = roc_curve(np.array(Label_VD),np.array(ResNet_prob_VD)) ###计算真正率和假正率
    auc_VD = auc(fpr,tpr)
    print('VD loss %2.4f' % (np.mean(test_loss)))
    print('VD Accuracy %2.4f' % (acc))
    print('VD AUC:%.3f'%auc_VD)
        
    if epoch % save_freq == 0 or auc_VD>0.73 :            
        # state_dict = net.state_dict()
        # for key in state_dict.keys():
        #     state_dict[key] = state_dict[key].cpu()
            
        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),},
            os.path.join(save_dir, 'Task2_ResNet_%03d.ckpt' % epoch))
                      
    endtime = time.time()
#    metrics = np.asarray(metrics, np.float32)
#    acc = np.asarray(acc, np.float32)

    print('time:%3.2f'%(endtime-starttime))
    
if __name__ == '__main__':
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    
    res_net = ClassifyNet().cuda()

    optimizer = torch.optim.Adam(res_net.parameters(), 
        lr=0.003, weight_decay = 1e-4)
    
    loss = torch.nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True

    dataset = GGODataGenerator(phase='train')
    dataset_VD = VD_DataGenerator()

#    indices = torch.randperm(len(dataset)).tolist()
#    dataset = torch.utils.data.Subset(dataset, indices[:-100])
#    dataset_test = torch.utils.data.Subset(dataset_test, indices[-100:])
    
    # define training and validation data loaders
    data_loader = DataLoader(
        dataset, batch_size=256, shuffle=True, num_workers=0, pin_memory=True)

    data_loader_VD = DataLoader(
        dataset_VD, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    

    
    
    save_freq = 100
    epochs = 500
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=100,
                                               gamma=0.1)
    save_dir = './model'
    for epoch in range(0, epochs + 1):
        train(data_loader, data_loader_VD, res_net, loss, epoch, optimizer,lr_scheduler, save_freq, save_dir)
