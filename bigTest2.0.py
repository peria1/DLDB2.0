#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 10:40:58 2018

I hacked this from my Oct 24 code that made a full tissue segment display of 
the output of the cancer detector, rather than showing only one tile at a time. 


@author: bill
"""

##!/usr/bin/env python3
## -*- coding: utf-8 -*-
#"""
#Created on Wed Oct 24 08:58:43 2018
#
#@author: bill
#"""

from model_inspect import Model
import openslide
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import billUtils as bu
from matplotlib.backends.backend_pdf import PdfPages
import sys
import dldb
from dldb import dlTile



#def grab_new_batch(N=None, maskfile = None, augment = False, boundary_kernel=None):
#
#    if N == None:
#        N=list(np.random.randint(0,size=batch_size,high=1260))
#    
#    if maskfile == None:
#        maskfile='test3_cancer.tif'
#
#    indata,y = db.feed_pytorch(N=N,maskfile=maskfile, augment=augment)
#    y = (y - np.min(y))/(np.max(y)-np.min(y))
#    
##        y = y[:,0:2,:,:]
##        y[:,1,:,:] = 1 - y[:,1,:,:]
##        y = torch.from_numpy(y).float()
#
#    if (type(criterion)==torch.nn.modules.loss.BCELoss)| \
#        (type(criterion)==torch.nn.modules.loss.BCEWithLogitsLoss):
#
#        y = y[:,0,:,:]
#        y = 1-y
#      #  y[:,1,:,:] = 1 - y[:,1,:,:]
#        y = torch.from_numpy(y).float()
#    elif type(criterion)==torch.nn.modules.loss.CrossEntropyLoss:
#        y = y[:,0,:,:]
#        y = torch.from_numpy(y).long()
#    else:
#        print('Unknown loss function type')

#        y = y.unsqueeze(1)
        
#    if boundary_kernel is not None:
#        with torch.no_grad():
#            npad = int((boundary_kernel.size()[2]-1)/2)
#            pad = torch.nn.ReplicationPad2d(npad)
#            ys = y.shape        
#            y = torch.Tensor.view(y,(ys[0],1,ys[1],ys[2]))
#            y = torch.nn.functional.conv2d(pad(y) ,boundary_kernel)
#            y = torch.Tensor.view(y,ys)
#            y = torch.round(y*2)/2
#        
#    if GPU:
#        indata = indata.cuda()
#        y = y.cuda()
#    return indata,y

if __name__ == "__main__":

#    pth = \
#'/media/bill/Windows1/Users/peria/Desktop/work/Brent Lab/Boucheron CNNs/' + \
#'DLDBproject/DLDB_20181015_0552'
#
#    db = dldb.DLDB(pth)
    
    batch_size, n_class, h, w = 20, 1, 256, 256

    show_plots = False
    GPU = True
    pretrained = False
    
    DLDBdir = '/media/bill/Windows1/Users/peria/Desktop/work/Brent Lab/' + \
    'Boucheron CNNs/DLDBproject/'

    
#    vggname = DLDBdir + 'vgg20181024_0253'  # first useful cancer detector
#    fcnname = DLDBdir + 'FCN20181024_0253' 
    vggname = DLDBdir + 'vgg20181205_2144'  # cancer detector with per color normalization
    fcnname = DLDBdir + 'FCN20181205_2144'
         
    m = Model(batch_size=batch_size, GPU=GPU, fcn_name=fcnname, vggname=vggname,\
              n_class=n_class)
    fcn_model = m.net.cuda().eval()
    
    
    file_to_process = '../NickReder/test4_annotated.tif'
    maskfile = '../NickReder/test4_cancer.tif'
    
#    file_to_process = 'NickRederRawData/18-040_n7_Z001981.tif'
#    file_to_process = 'NickRederRawData/18-040_n7_Z001985.tif'
#    file_to_process = 'NickRederRawData/18-040_n7_Z001989.tif'
#    maskfile = None
    
    openslide.Image.MAX_IMAGE_PIXELS = None # prevents DecompressionBomb Error
    Tissue = openslide.open_slide(file_to_process)
    
    allgs = None
    if maskfile:
        gs =  openslide.open_slide(maskfile)
        allgs = np.asarray(gs.read_region((0,0),0,gs.dimensions))[:,:,0]
        allgs = 1.0 - allgs/np.max(allgs)

#    tsize = 256
#    origin = (34643,309)
#    
#    all_tiles = [i for i in range(2495, 3653)]
#    istart = 0
#    iend = len(all_tiles)
    
#    allgs = np.asarray(gs.read_region((0,0),0,gs.dimensions))
    
    allT = np.asarray(Tissue.read_region((0,0),0,Tissue.dimensions))[:,:,0:3]
#    T0 = np.mean(allT,axis=(0,1),dtype=np.float32)
#    dT = np.std(allT,axis=(0,1),dtype=np.float32)
    T0 = np.float32(np.mean(allT,axis=(0,1)))
    dT = np.float32(np.std(allT,axis=(0,1)))
    
    test_out = np.zeros_like(allT,dtype=np.float64)[:,:,0]
    
    nx, ny = 4096, 256  # nx and ny need to be powers of two. 4096 x 256 maxes out CUDA RAM
#    nx, ny = 256, 256
    NX, NY = Tissue.dimensions
    IX, IY = (0,0)
    with torch.no_grad():
        while IY < NY:
            IX = 0
            h = ny 
            while (IY + h > NY): 
                print('dividing h...')
                h//=2 
            while IX < NX:
                w = nx
                while IX + w > NX:
                    print('dividing w...')
                    w//=2
                chnk = np.asarray(Tissue.read_region((IX,IY),0,(w,h)),\
                                        dtype=np.float32)[:,:,0:3]
                chnk = (chnk - T0)/dT # T0 and dT have 3 elements, so they broadcast
                chnk = np.transpose(chnk,axes=[2,0,1])
                chnkfeed = torch.tensor(chnk).unsqueeze(0).cuda()
                outchnk = \
                fcn_model(chnkfeed).cpu().detach()
                test_out[IY:(IY+h),IX:(IX+w)] = outchnk
                
                IX += nx
            IY += ny
            print(IY, chnkfeed.shape)
                  
    fakegs = np.zeros_like(test_out)
    fakegs[test_out >= 0.67] = 1

    ncols = 1
    if maskfile:
        nrows = 3
        ifake = 2
        itrue = 3
    else:
        nrows = 2
        ifake = 2
        itrue = None
        

    plt.figure()
    plt.subplot(nrows,ncols,1)
    plt.subplots_adjust(wspace=0, hspace=0)
#    f, axarr = plt.subplots(nrows,ncols, gridspec_kw = {'wspace':0, 'hspace':0})
#    axarr[0].imshow(allT)
#    axarr[1].imshow(fakegs)
#    axarr[2].imshow(gs)
    
    plt.imshow(allT)
    plt.subplot(nrows,ncols,ifake)
    plt.imshow(fakegs)    
    plt.title(' '.join([bu.just_filename(bu,fcnname),'with',bu.just_filename(bu,file_to_process)]))
    if allgs is not None:
        plt.subplot(nrows, ncols, itrue)
        plt.imshow(allgs)
        plt.title(bu.just_filename(bu,maskfile))

#    tile = Tissue.read_region((34643,309),0,(tsize,tsize))
#    tile = np.asarray(tile)
#    tile = np.transpose(tile,axes=(2,0,1))
#    tile = tile[0:3,:,:]
#    tile = np.reshape(tile,(1,3,tsize,tsize))
#    tile = (tile - np.mean(tile))/np.std(tile)
#    
#    out = torch.sigmoid(fcn_model(torch.tensor(tile,dtype=torch.float32).cuda()))
#    out = out.cpu().detach().numpy()
#    
#    
    
#    plt.figure(2)
#    plt.clf()
#    plt.imshow(gs.read_region(origin,0,(tsize,tsize)))
#    
#    
#    plt.figure(3)
#    plt.clf()
#    plt.imshow(out)
#    plt.colorbar()