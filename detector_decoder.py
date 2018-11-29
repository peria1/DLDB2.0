#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 14:01:30 2018

Here I am trying to turn my cancer detector into an autoencoder by freezing the encoder
and training a decoder to reproduce the input. So, the cancer decision has been made, 
and I want to be able to zero out features that activate strongly in the presence 
of cancer, to see how that changes the images. WJP 3-Nov-2018

@author: bill
"""
# -*- coding: utf-8 -*-
#
# Copied from github: https://github.com/pochih/FCN-pytorch/blob/master/python/fcn.py
#
#  This is the pared down and evolved version of FCNpytorchFromGithub.py
#
#
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
#import billUtils as bu
#from matplotlib.backends.backend_pdf import PdfPages
#import sys
import dldb
from dldb import dlTile
import fcn

#----------------------
   

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
    
#----------------------


def grab_new_batch(N=None, maskfile = None, augment = False, boundary_kernel=None):

    if N == None:
        N=list(np.random.randint(0,size=batch_size,high=1260))
    

    indata = db.feed_pytorch(N=N)
    
    if GPU:
        indata = indata.cuda()
    return indata

#----------------------

if __name__ == "__main__":
    
    show_plots = False
    GPU = True

    pth = \
'/media/bill/Windows1/Users/peria/Desktop/work/Brent Lab/Boucheron CNNs/' + \
'DLDBproject/DLDB_20181015_0552'

    db = dldb.DLDB(pth)
    
    batch_size, n_class, h, w = 20, 3, 256, 256

    fcn_model = fcn.load_model(GPU=GPU, n_class=n_class,load_decoder=False)
    
    criterion = nn.MSELoss() 
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-3, momentum=0.9)
    
    saveloss = []

    if GPU:
        fcn_model = fcn_model.cuda()

    params_to_freeze = [item for key,item in fcn_model.named_parameters() if "pretrained" in key]
    for p2f in params_to_freeze:
        p2f.requires_grad = False

#--------------------------TRAIN    

    indata = grab_new_batch()

    early = False
    fcn_model.train()
    for iteration in range(50000):
        optimizer.zero_grad()
        output = fcn_model(indata)
        if iteration == 0:
            if GPU:
                output = output.cuda()

        with torch.enable_grad():
            loss = criterion(output,indata)
        

        count = 0
        while (torch.isnan(loss) and count < 10):
            print('ARGH! Loss is NaN...trying new data...')
            indata = grab_new_batch()
            count+=1
            output = fcn_model(indata)
            loss = criterion(output,indata)


        if count >= 10:
            break
        
        loss.backward()
        saveloss.append(loss.item())
        optimizer.step()
        
        if iteration % 20 == 0:
            torch.save(fcn_model.state_dict(),'preFCNcurrent')

            if len(saveloss) <= 20:
                print("iteration {}, loss {:.3f}".format(iteration, loss.item()))
            else:
                print("{:d} max loss in last 20 is {:.3f}".format(len(saveloss)-1,np.max(saveloss[-20:-1])) )
                ochk = output.cpu().detach().numpy()
            
            if show_plots:
                plt.clf()
                plt.plot(saveloss)
                xl = plt.xlim()
                plt.hlines(0.693,xl[0],xl[1]); # Need to be uniformly better than this!
                plt.pause(0.1)
                
        if not early:
            indata = grab_new_batch()
            
            
    torch.save(fcn_model.state_dict(),'preFCN' + db.date_for_filename())
        
        


        
        
        
        
        
        
        
        
        
        
        
        
        