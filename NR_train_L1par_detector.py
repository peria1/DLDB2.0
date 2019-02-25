#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 15:39:37 2019

@author: bill
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
import numpy as np
import matplotlib.pyplot as plt
import billUtils as bu
import sys

#from vggfcn import VGGNet, FCN8s
import fcn

#from matplotlib.backends.backend_pdf import PdfPages
#import sys

import dldb
from dldb import dlTile

import copy
#
#
#------------------------------------------

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
        plt.subplot(1,2,2)
        plt.imshow(m[i,:,:])
        plt.title(str(i))
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
    return indata,y

#----------------------
def reg_loss(model, plist=None):
    regularization_loss = torch.tensor(0.).type(torch.float).cuda()
    if plist is None:
        for name, param in model.named_parameters():
            ok = 'pretrained_net' in name and 'bias' not in name
            if ok:
                regularization_loss += torch.sum(torch.abs(param))
    else:
        for p in plist:
            regularization_loss += torch.sum(torch.abs(p))
    
    return regularization_loss

if __name__ == "__main__":
    
    itmax = 50000
    L1_alpha = 1e-4
    
    L1_pnamelist = ['pretrained_net.features.24.weight',\
                    'pretrained_net.features.26.weight',\
                    'pretrained_net.features.28.weight']
    
    bd = 41 # boundary_distance This is the typical error expected when a pathologist 
            # draws a boundary, measured in pixels. 
            
    ck = torch.reshape(torch.tensor(circle_kernel(bd)).type(torch.float),(1,1,bd,bd)) 
#    ck = None
#    print('turned off circle kernel boundary ignorer...')
    
    show_plots = False
    GPU = True
    pretrained = False

#    pth = \
#'/media/bill/Windows1/Users/peria/Desktop/work/Brent Lab/Boucheron CNNs/' + \
#'DLDBproject/DLDB_20180827_0753'
#    pth = \
#'/media/bill/Windows1/Users/peria/Desktop/work/Brent Lab/Boucheron CNNs/' + \
#'DLDBproject/DLDB_20181015_0552'

    pth = bu.uichoosedir(title='Choose a DLDB...')

    db = dldb.DLDB(pth)
    
    batch_size, n_class, h, w = 20, 1, 256, 256

    fcn_name = None

    fcn_model = fcn.load_model(n_class=n_class,fcnname=fcn_name,\
                               freeze_encoder=False, load_decoder=False)
    
    plist = []
    for n,p in fcn_model.named_parameters():
        if 'bn3.weight' in n:
            bn3 = p
        if n in L1_pnamelist:
            plist.append(p)

    reload = 'reload' in sys.argv
    if reload:
        print('using FCNcurrent...')
        fcn_model.load_state_dict(torch.load('FCNcurrent'))
    
    
    pw = torch.as_tensor(8.).type(torch.float).cuda()
#    pw = torch.as_tensor(1.).type(torch.float).cuda()  # needed for lstar

    criterion = nn.BCEWithLogitsLoss(pos_weight = pw,reduction='none')    
#    criterion = nn.MSELoss()
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-3, momentum=0.9)
    
    saveloss = []

    if GPU:
        fcn_model = fcn_model.cuda()

#--------------------------TRAIN    

    indata,y = grab_new_batch(augment=True, boundary_kernel=ck)

#    print('no boundary kernel...')

    early = not reload
    
    fcn_model = fcn_model.train()
    params = [p for p in fcn_model.parameters()]
    fcn_no_nans = copy.deepcopy(fcn_model)
    for iteration in range(itmax):
        optimizer.zero_grad()
        output = fcn_model(indata)
        if torch.sum(torch.isnan(output)) > 0:
            print('output has NaNs...')
    #       output = torch.sigmoid(output) # needed for use in plain BCELoss, no logits. 
        if iteration == 0:
            if GPU:
                output = output.cuda()

        with torch.enable_grad():
            pixloss = criterion(output,y)
            lst = lstar(output,y,pw)
            c2 = torch.mean(pixloss + lst)
            
            regularization_loss = reg_loss(fcn_model, plist)                
            loss = c2 + L1_alpha * regularization_loss
        
        count = 0
        while (torch.isnan(loss) and count < 10):
            print('ARGH! Loss is NaN... reloading model state and trying new data for iteration',iteration,'...')
            print('c2:',torch.mean(c2),'lst:',torch.mean(lst),'reg:',regularization_loss)
            print('bn3 mean:',torch.mean(bn3))

            indata,y = grab_new_batch(augment=True,boundary_kernel=ck)
            optimizer.zero_grad()
            
            fcn_model = fcn.load_model(n_class=n_class,fcnname=fcn_name,\
                                       freeze_encoder=False, load_decoder=False)
            fcn_model.load_state_dict(torch.load('FCNcurrent'))
            fcn_model = fcn_model.cuda().train()
            output = fcn_model(indata)
            
            with torch.enable_grad():
                pixloss = criterion(output,y)
                lst = lstar(output,y,pw)
                c2 = torch.mean(pixloss + lst)
                
                regularization_loss = reg_loss(fcn_model,plist)                
                loss = c2 + L1_alpha * regularization_loss
                count+=1
    
    
        if count >= 10:
            print('More than 10 batches yielded NaNs, bailing...')
            torch.save(fcn_model.state_dict(),'FCNbail' + db.date_for_filename())
            break
#        
        loss.backward()
        saveloss.append(loss.item())
        some_nans = False
        optimizer.step()
        for n, p in fcn_model.named_parameters():
            nnan = torch.sum(torch.isnan(p))
            if nnan > 0:
                some_nans = True
                print(n,'has',nnan,'NaNs...')
        
        if some_nans:
            print('Going back to previous NaN-free model...')
            fcn_model = copy.deepcopy(fcn_no_nans)
        else:
            fcn_no_nans = copy.deepcopy(fcn_model)
        
        
        if c2 < 0.5 and early: 
            early = False           
            print('setting early to false...') 
        
        if iteration % 20 == 0:
            torch.save(fcn_model.state_dict(),'FCNcurrent')
#            torch.save(vgg_model.state_dict(),'VGGcurrent')
            print('c2:',torch.mean(c2).item())
            print('L1loss:',regularization_loss.item() * L1_alpha)
            print('bn3 mean:',torch.mean(bn3))
            print('\n')

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
            
            
    torch.save(fcn_model.state_dict(),'FCN' + db.date_for_filename())
#    torch.save(vgg_model.state_dict(),'vgg' + db.date_for_filename())
        
        


        
        
        
        
        
        
        
        
        
        
        
        
        
