#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:59:48 2018

@author: bill
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import billUtils as bu
import dldb
from dldb import dlTile

from matplotlib.backends.backend_pdf import PdfPages
import sys

class FCN8s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.drop    = nn.Dropout2d(p = 0.5)
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)

        score = self.relu(self.deconv1(x5))               # size=(N, 512, x.H/16, x.W/16)
        score = self.drop(score)
        score = self.bn1(score + x4)                      # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv2(score))            # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x3)                      # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score.squeeze()  # size=(N, n_class, x.H/1, x.W/1)

class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, \
                 remove_fc=True, show_params=False, GPU = False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.device)
        if GPU:
            for name, param in self.named_parameters():
                param.cuda()
                
 

    def forward(self, x):
        output = {}

        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x

        return output


ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

#------------------------------------------
def show_batch(d,m,nn=None):
    n = d.shape[0]
    
    if nn==None:
        nn=n
        
    d = np.transpose(d.detach().cpu().numpy(),axes=(0,2,3,1))
    d = (d-np.min(d))/(np.max(d)-np.min(d))
    m = m.detach().cpu().numpy()
    for i in range(nn):
        plt.clf()
        plt.subplot(1,2,1)
        plt.imshow(d[i,:,:,:])
        plt.subplot(1,2,2)
        plt.imshow(m[i,:,:])
        plt.pause(0.5)
    
#----------------------grab_new_batch


def grab_new_batch(N=None, maskfile = None, augment = False, boundary_kernel=None):

    if N == None:
        N=list(np.random.randint(0,size=batch_size,high=1260))
    
    if maskfile == None:
        maskfile='test3_cancer.tif'

    indata,y = db.feed_pytorch(N=N,maskfile=maskfile, augment=augment)
       
    if boundary_kernel is not None:
        with torch.no_grad():
            npad = int((boundary_kernel.size()[2]-1)/2)
            pad = torch.nn.ReplicationPad2d(npad)
            ys = y.shape   
            y = torch.Tensor.view(y,(ys[0],1,ys[1],ys[2]))
            y = torch.nn.functional.conv2d(pad(y).type(torch.float), boundary_kernel)
            y = torch.Tensor.view(y,ys)
            y = torch.round(y*2)/2
    
    if GPU:
        indata = indata.cuda()
        y = y.cuda()
    return indata,y



#----------------------
#
#def grab_new_batch(N=None, maskfile = None, augment = False):

#    if N == None:
#        N=list(np.random.randint(0,size=batch_size,high=1260))
    
#    if maskfile == None:
#        maskfile='test3_cancer.tif'

#    indata,y = db.feed_pytorch(N=N,maskfile=maskfile, augment=augment)
#    y = (y - np.min(y))/(np.max(y)-np.min(y))
    
#        y = y[:,0:2,:,:]
#        y[:,1,:,:] = 1 - y[:,1,:,:]
#        y = torch.from_numpy(y).float()

#    if (type(criterion)==torch.nn.modules.loss.BCELoss)| \
#        (type(criterion)==torch.nn.modules.loss.BCEWithLogitsLoss):

#        y = y[:,0,:,:]
#        y = 1-y
      #  y[:,1,:,:] = 1 - y[:,1,:,:]
#        y = torch.from_numpy(y).float()
#    elif type(criterion)==torch.nn.modules.loss.CrossEntropyLoss:
#        y = y[:,0,:,:]
#        y = torch.from_numpy(y).long()
#    else:
#        print('Unknown loss function type')

#        y = y.unsqueeze(1)
#    if GPU:
#        indata = indata.cuda()
#        y = y.cuda()
#    return indata,y

#----------------------

if __name__ == "__main__":
    
    import billUtils as bu


 #   pth = '/home/bill/DLDBproject/DLDB_20181010_1703'
 #   pth = bu.uichoosedir()

 #   pth = '/home/bill/DLDBproject/DLDB_20181012_1552' 

    pth = bu.uichoosedir()

    db = dldb.DLDB(pth)
    
    batch_size, n_class, h, w = 20, 1, 256, 256


    show_plots = False
    GPU = True
    pretrained = False
    reload = True


    
    if not pretrained:
        print('using untrained VGG...')
        
    vgg_model = VGGNet(pretrained = pretrained, requires_grad=True, GPU = GPU)
    fcn_model = FCN8s(pretrained_net=vgg_model, n_class=n_class)


    if reload:
#        vgg_model.load_state_dict(torch.load(bu.uichoosefile()))
#        fcn_model.load_state_dict(torch.load(bu.uichoosefile()))
        print('using VGGcurrent and FCNcurrent...')
        vgg_model.load_state_dict(torch.load('VGGcurrent'))
        fcn_model.load_state_dict(torch.load('FCNcurrent'))
       
    criterion = nn.BCEWithLogitsLoss(pos_weight = torch.as_tensor(8.).cuda())    
#    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-3, momentum=0.9)
    

    if GPU:
        fcn_model = fcn_model.cuda()

#--------------------------TEST    

        
Nlist = list(np.random.randint(2495,size=20,high=3652))
intest, ytest = grab_new_batch(N=Nlist, maskfile = 'test4_cancer.tif')

fcn_model.eval()
outtest = torch.sigmoid(fcn_model(intest))
ochk = outtest.cpu().detach().numpy()
ychk = ytest.cpu().detach().numpy()
acc = np.average(np.abs(np.round(ochk) - ychk))
print("{:d} % correct".format(int(acc*100)))
print(criterion(outtest,ytest))

pp = PdfPages('test' + bu.date_for_filename() +'.pdf')
plt.ioff()

tiles = db.get_tile_by_number(Nlist)
for i,n in enumerate(Nlist):
    plt.clf()
#    tile = db.get_tile_by_number(n)[0]
    tile = tiles[i]
#    plt.figure(2)
#    plt.clf()
    plt.subplot(2,2,1)
    tile.show()
    plt.pause(0.1)

#    plt.figure(3)
#    plt.clf()
    plt.subplot(2,2,2)
    plt.imshow(ychk[i,:,:],vmin=0,vmax=1)
    plt.title('pathologist')
    plt.colorbar()
    plt.pause(0.1)
    
    plt.subplot(2,2,3)
    plt.imshow(ychk[i,:,:],vmin=0,vmax=1)
    plt.title('pathologist')
   # plt.colorbar()  # this gets in the way of comparing vertically
    plt.pause(0.1)

#    plt.subplot(2,2,3)
#    plt.imshow(disp[i,:,:],vmin=0,vmax=1)
#    plt.title('bottleneck')
   # plt.colorbar()  # this gets in the way of comparing vertically
    plt.pause(0.1)

#    plt.figure(4)
#    plt.clf()
    plt.subplot(2,2,4)
    plt.imshow(ochk[i,:,:],vmin=0,vmax=1)
    plt.title('trained model')
    plt.colorbar()
    plt.pause(1.0)

    pp.savefig(bbox_inches='tight')
    
pp.close()    
        
        
        
        
        
        
        
        
        
        
        
        
