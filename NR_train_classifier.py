#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 09:57:29 2019

@author: bill
"""
#
# Copied from github: https://github.com/pochih/FCN-pytorch/blob/master/python/fcn.py
#
#  This is the pared down and evolved version of FCNpytorchFromGithub.py
#
#

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
import numpy as np
import matplotlib.pyplot as plt
import billUtils as bu
import sys

from vggfcn import VGGNet, FCN8s
#import fcn

#from matplotlib.backends.backend_pdf import PdfPages
#import sys

import dldb
from dldb import dlTile
#
#
#------------------------------------------
def tile_has_cancer(tiles):
    return 


def lstar(z,y,pw):
    
    zero = torch.zeros_like(z)
    one = torch.ones_like(z)

    logp = torch.log(torch.pow(torch.sigmoid(z),pw)*(1-torch.sigmoid(z)))
    
    logp[y==zero] = 0
    logp[y== one] = 0
    
    return 2*y*(1-y)*logp

#----------------------
def show_batch(d,m,nn=None, delay = 0.5):
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
        plt.title(str(i))
        plt.subplot(1,2,2)
        plt.imshow(m[i,:,:])
        plt.title(str(np.mean(m[i,:,:])))
        plt.pause(delay)
        
#----------------------
def circle_kernel(N):
    if N % 2 is not 1:
        print('kernel size must be odd...')
        return None
    
    rmax = (N-1)/2.0
    x = np.linspace(-rmax,rmax,num=N)
    xx = np.reshape(np.kron(np.power(x,2),np.ones_like(x)),(N,N))
    r = np.sqrt(xx + np.transpose(xx))
    ck = r < rmax
    ck = ck/np.sum(ck)

    return ck

#----------------------

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
        
    
    return indata, y

#----------------------

if __name__ == "__main__":
    
    bd = 41 # boundary_distance This is the typical error expected when a pathologist 
            # draws a boundary, measured in pixels. 
            
    ck = torch.reshape(torch.tensor(circle_kernel(bd)).type(torch.float),(1,1,bd,bd)) 
#    ck = None
#    print('turned off circle kernel boundary ignorer...')
    
    show_plots = True
    GPU = True
    pretrained = False

#    pth = \
#'/media/bill/Windows1/Users/peria/Desktop/work/Brent Lab/Boucheron CNNs/' + \
#'DLDBproject/DLDB_20180827_0753'
#    pth = \
#'/media/bill/Windows1/Users/peria/Desktop/work/Brent Lab/Boucheron CNNs/' + \
#'DLDBproject/DLDB_20181015_0552'

    pth = bu.uichoosedir(title='Choose DLDB...')

    db = dldb.DLDB(pth)
    
    batch_size, n_class, h, w = 20, 1, 256, 256
    
    cancer_threshold = 0.05
 
    if not pretrained:
        print('using untrained VGG...')
        
#"""
## Load the pretrained model from pytorch
#vgg16 = models.vgg16_bn()
#vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))
#print(vgg16.classifier[6].out_features) # 1000 
#
#
## Freeze training for all layers
#for param in vgg16.features.parameters():
#    param.require_grad = False
#
## Newly created modules have require_grad=True by default
#num_features = vgg16.classifier[6].in_features
#features = list(vgg16.classifier.children())[:-1] # Remove last layer
#features.extend([nn.Linear(num_features, len(class_names))]) # Add our layer with 4 outputs
#vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
#print(vgg16)
#"""
#    vgg_model = VGGNet(pretrained = pretrained, requires_grad=True, \
#                       remove_fc = False, GPU = GPU)
    vgg_model = models.vgg16()
    vgg_model.classifier[0] = nn.Linear(in_features=32768, \
                                        out_features=4096,bias=True)
    vgg_model.classifier[6] = nn.Linear(in_features=4096, \
                                        out_features=1, bias=True)
    
    vgg_model = vgg_model.cuda()

    
    reload = 'reload' in sys.argv
    if reload:
#        vgg_model.load_state_dict(torch.load(bu.uichoosefile()))
#        fcn_model.load_state_dict(torch.load(bu.uichoosefile()))
        print('using VGGcurrent and FCNcurrent...')
        vgg_model.load_state_dict(torch.load('VGGcurrent'))
    
    
#    pw = torch.as_tensor(8.).type(torch.float).cuda()
#    criterion = nn.BCEWithLogitsLoss(pos_weight = pw,reduction='none')    
    criterion = nn.BCEWithLogitsLoss()    
    optimizer = optim.SGD(vgg_model.parameters(), lr=1e-3, momentum=0.9)
    
    saveloss = []

    if GPU:
        vgg_model = vgg_model.cuda()

#--------------------------TRAIN    

    indata,y = grab_new_batch(augment=True, boundary_kernel=ck)

#    print('no boundary kernel...')

    early = not reload
    
    vgg_model.train()
    for iteration in range(4000):
        
        optimizer.zero_grad()
        output = torch.squeeze(vgg_model(indata))
    #       output = torch.sigmoid(output) # needed for plain BCELoss, no logits
        if iteration == 0:
            if GPU:
                output = output.cuda()
#        loss = criterion(output, y)
                
        with torch.enable_grad():
            ym = torch.mean(torch.mean(y,-1),-1)
            
#            loss = torch.mean(\
#                      criterion(output,\
#                           torch.tensor([(1 if yy > cancer_threshold else 0) \
#                                         for yy in ym],\
#                                            dtype=torch.float).cuda()))
            loss = criterion(output,\
                           torch.tensor([(1 if yy > cancer_threshold else 0) \
                                         for yy in ym],\
                                            dtype=torch.float).cuda())
        
        count = 0
        while (torch.isnan(loss) and count < 10):
            print('ARGH! Loss is NaN...trying new data...')
            indata,y = grab_new_batch(augment=True,boundary_kernel=ck)
            count+=1
            output = vgg_model(indata)
            c2 = criterion(output,y)
            lst = lstar(torch.sigmoid(output),y,pw)
            pixloss = c2 + lst
            loss = torch.mean(pixloss)


        if count >= 10:
            break
        
        loss.backward()
        saveloss.append(loss.item())
        optimizer.step()
        
        if loss.item() < 0.5 and early: 
            early = False           
            print('setting early to false...') 
        
        if iteration % 20 == 0:
            torch.save(vgg_model.state_dict(),'VGGcurrclass')

            if len(saveloss) <= 20:
                print("iteration {}, loss {:.3f}".format(iteration, loss.item()))
            else:
                print("{:d} max loss in last 20 is {:.3f}".format(len(saveloss)-1,np.max(saveloss[-20:-1])) )
                ochk = output.cpu().detach().numpy()
                ychk = y.cpu().detach().numpy()
#                acc = np.average(np.round(ochk) == ychk)
#                print("{:d} % correct".format(int(acc*100)))
            
            if show_plots:
                plt.clf()
                plt.plot(saveloss)
                xl = plt.xlim()
                plt.hlines(0.693,xl[0],xl[1]); # Need to be uniformly better than this!
                plt.pause(0.1)

            if not early:
                indata,y = grab_new_batch(augment=True, boundary_kernel=ck)
            
            
    torch.save(vgg_model.state_dict(),'vggclass' + db.date_for_filename())
        
        


        
        
        
        
        
        
        
        
        
        
        
        
        
