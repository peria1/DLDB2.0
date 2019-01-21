#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 07:38:47 2018

@author: bill
"""

import billUtils as bu
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.vgg import VGG
import copy

class FCN8s(nn.Module):

    def __init__(self, pretrained_net, n_class): # , drop_layer=drop_layer):
        super().__init__()
        
#        self.drop = None
#        if drop_layer:
#            self.drop    = nn.Dropout2d(p = 0.5)
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
#        if self.drop is not None:
#            score = self.drop(score)
#        else:
#            print('No drop layer...')
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

def load_model(GPU=True,n_class=1,load_encoder=False,load_decoder=True,\
               vggname=None, fcnname=None, \
               requires_grad = True, freeze_encoder = False): #, drop_layer=True):
    
    if requires_grad == freeze_encoder:
        print('requires_grad is',requires_grad,', freeze_encoder is ',freeze_encoder,'...this is inconsistent.')

    requires_grad = requires_grad or not freeze_encoder
    # Get the structure of VGG. I don't want to use their pre-trained model (ImageNet?)
    vgg_model = VGGNet(pretrained = False, requires_grad=requires_grad, GPU = GPU)
    
    # Get the structure of FCN8 decoder. 
    fcn_model = FCN8s(pretrained_net=vgg_model, n_class=n_class) 
    
#    if vggname is None:
##        vggname = '/media/bill/Windows1/Users/peria/Desktop/work/Brent Lab/Boucheron CNNs/DLDBproject/vgg20181017_0642'
#        vggname = bu.uichoosefile(title='Choose VGG file...')
#    if fcnname is None:
##        fcnname = '/media/bill/Windows1/Users/peria/Desktop/work/Brent Lab/Boucheron CNNs/DLDBproject/FCN20181017_0642'
#        fcnname = bu.uichoosefile(title='Choose FCN file...')
        
    if load_encoder:
        if vggname is None:
            vggname = bu.uichoosefile(title='Choose VGG file...')
        print('Loading encoder state from '+bu.just_filename(bu,vggname)+'...')
        vgg_model.load_state_dict(torch.load(vggname))
    else:
        print('Not loading encoder state...')

    if load_decoder:   # Note that loading the state_dict here will overwrite 
                       # any state that is already in the encoder.
        if fcnname is None:
            fcnname = bu.uichoosefile(title='Choose FCN file...')
        print('Loading decoder state from '+bu.just_filename(bu,fcnname)+'...')
        fcn_model.load_state_dict(torch.load(fcnname))
    else:
        print('Not loading decoder state...')

    if load_encoder:   # if True, this will copy the encoder from vgg_model into fcn_model
        new_dict = copy.deepcopy(fcn_model.state_dict())
        for k in vgg_model.state_dict().keys():
            if 'pretrained' in k:
                new_dict[k] = copy.deepcopy(vgg_model.state_dict()[k])
        
        fcn_model.load_state_dict(new_dict)        
               
        
        
    return fcn_model
