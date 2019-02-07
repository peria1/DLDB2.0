#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 06:19:12 2019

@author: bill
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:14:58 2019

@author: bill
"""


import numpy as np
import billUtils as bu

import dldb
from dldb import dlTile

import fcn
import torch
import matplotlib.pyplot as plt

import copy
import gc

FEATURE_MAPS = None #'nothing to see here yet'  # a global to hold maps obtained via hook functions. 


n_class = 1
GPU = True
batch_size = 20

dldb_path = \
'/media/bill/Windows1/Users/peria/Desktop/work/Brent Lab/'+\
'Boucheron CNNs/DLDBproject/DLDB_20181015_0552'

db = dldb.DLDB(input_directory = dldb_path)


fcn_name = '/media/bill/Windows1/Users/' + \
                  'peria/Desktop/work/Brent Lab/Boucheron CNNs/'+\
                  'DLDBproject/FCN20181205_2144'
#fcn_name = '/media/bill/Windows1/Users/' + \
#                  'peria/Desktop/work/Brent Lab/Boucheron CNNs/'+\
#                  'DLDBproject/FCN20190122_0700'
#fcn_name = '/media/bill/Windows1/Users/' + \
#                  'peria/Desktop/work/Brent Lab/Boucheron CNNs/'+\
#                  'DLDBproject/redo_translate_for_16520190121_0825'

net = fcn.load_model(n_class=n_class,fcnname=fcn_name).eval().cuda()

for p in net.parameters():
    p.requires_grad = False


def grab_new_batch(N=None, maskfile = None, augment = False, boundary_kernel=None):

    if not maskfile:
        maskfile = 'test3_cancer.tif'
        
#    print('Fixed N!')
#    N = [ 968,  521 , 419 , 737 ,1140, 1240,  677, 1178,  694,  255, \
#     281,  451, 1182,  527,  203,  156,  893,  398,  975, 1066]

    if N == None:
        N=list(np.random.randint(0,size=batch_size,high=1260))    

    indata, y = db.feed_pytorch(N=N, maskfile=maskfile,\
                                normalization='lump3')
    
    if GPU:
        indata = indata.cuda()
        if y is not None:
            y = y.cuda()
    
    return indata, y

def show_batch(d,m,nn=None, delay = 0.5):
#    import itertools
    
    def max_contrast(im):
        axes = tuple(range(1,len(im.shape)))
        im0 = np.min(im, axis=axes, keepdims=True)
        im1 = np.max(im, axis=axes, keepdims=True)
        denom = im1 - im0
        out = im - im0
        for i, dd in enumerate(denom):
            if not dd:
                denom[i] = 1.0
            
        return out/denom
                
    def trans(im):
        if type(im) is torch.Tensor:
            im = im.detach().cpu().numpy()
        ims = im.shape
        lms = len(ims)
        if lms > 3:
            axes = [0]
            for i in range(2,lms):
                axes.append(i)
            axes.append(1)
        else:
            axes = [i for i in range(lms)]
        print(axes)
        return max_contrast(np.transpose(im, axes = axes))

    d = trans(d)         
    m = trans(m)       
    n = d.shape[0]
    
    if nn==None:
        nn=n
        
    for i in range(nn):
        plt.clf()
        plt.subplot(1,2,1)
        plt.imshow(d[i,])
        plt.title(str(i))
        plt.subplot(1,2,2)
        plt.imshow(m[i,])
        plt.title(str(np.mean(m[i,])))
        plt.pause(delay)
        
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def accum_pdf_hook(self, input, output):
    global FEATURE_MAPS
#        
    pred = output.cpu().detach().numpy()
    
    nexamp, nfm, ny, nx = pred.shape
    pred = np.transpose(pred, axes=(0,2,3,1))
    pred = (pred-np.min(pred))/(np.max(pred)-np.min(pred))*128
    edges = np.arange(-255,255)
    
    i16 = np.int16(pred)
    dx = np.int16((np.roll(i16,-1,axis=2) - np.roll(i16,1,axis=2))/2.0)
    dy = np.int16((np.roll(i16,-1,axis=1) - np.roll(i16,1,axis=1))/2.0)
#    print(dx.shape)
#   The gradient shapes are (nexamp, ny, nx, nfm)

    dx = dx[:,1:-1,1:-1,:]
    dy = dy[:,1:-1,1:-1,:]
#    Oops now it's (nexamp, ny-2,nx-2,nfm)
#    H = np.zeros((nexamp,nfm,509,509))
    if isinstance(FEATURE_MAPS, type(None)):
        FEATURE_MAPS = np.zeros((nexamp,nfm,509,509),dtype=np.uint16)
        print('created zeros for FEATURE_MAPS...')

    for i in range(nexamp):
        for j in range(nfm):
            FEATURE_MAPS[i,j,:,:], xedges, yedges = \
                        np.histogram2d(dx[i,:,:,j].flatten(),\
                                       dy[i,:,:,j].flatten(),\
                                       bins=(edges, edges))

#    FEATURE_MAPS = H
    
#
        
    print(id(FEATURE_MAPS))
    
    return



if __name__ == '__main__':
    
    modules = [module for name, module in net.named_modules()\
                     if len(module._modules) == 0]
    mnames = [name for name, module in net.named_modules()\
                     if len(module._modules) == 0]
    pick = 29 # this yields pretrained_net 28
    print('Examining feature maps of',mnames[pick])

    ntry = 100
#    cpred = np.zeros(batch_size*ntry)
#    cpath = np.zeros_like(cpred)
#    Make bins for cpath and cpred. One bin is for exactly zero, then 10 more from 
#    0+ up to 1. 
    nnzbins = 10

    pmaps = None
    mmaps = None

    
    for i in range(ntry):
        print('Processing batch',i+1,'of',ntry)
        
        indata,y = grab_new_batch()

        hook = modules[pick].register_forward_hook(accum_pdf_hook)
        out = torch.sigmoid(net(indata))
        hook.remove()

        cpath = np.mean(np.mean(y.cpu().detach().numpy(),axis=-1),axis=-1)
        pbins = np.trunc(cpath*nnzbins) + 1
        pbins[cpath==0] = 0
        pbins[cpath==1] = nnzbins
        pbins = np.uint8(pbins)

        cpred = np.mean(np.mean((out.cpu().detach().numpy() > 0.9),axis=-1),axis=-1)
        mbins = np.trunc(cpred*nnzbins) + 1
        mbins[cpred==0] = 0
        mbins[cpred==1] = nnzbins
        mbins = np.uint8(mbins)

        if pmaps is None:
            pmaps = np.zeros((nnzbins+1,*FEATURE_MAPS.shape[1:]),dtype=np.uint16)
            mmaps = np.zeros((nnzbins+1,*FEATURE_MAPS.shape[1:]),dtype=np.uint16)
        
        for i in range(batch_size):
            pmaps[pbins[i],:,:,:] += FEATURE_MAPS[i,:,:,:]
            mmaps[mbins[i],:,:,:] += FEATURE_MAPS[i,:,:,:]


    nbins, nfm, ny, nx = pmaps.shape

    def entropy(C):
        not_zero = C > 0
        denom = np.sum(C)
        p = C[not_zero]/denom
        e = -np.sum(p*np.log2(p))
        return e 
    
    pentropy = np.zeros((nbins, nfm))
    mentropy = np.zeros((nbins, nfm))
    ppdf = np.zeros((nbins, nfm))
    mpdf = np.zeros((nbins, nfm))
    
    ent = np.zeros((nbins,nfm))
    for i in range(nbins):
        for j in range(nfm):
            pentropy[i,j] = entropy(pmaps[i,j,:,:])
            mentropy[i,j] = entropy(mmaps[i,j,:,:])
#            ppdf[i,j] = pmaps[i,j,:,:]/np.sum(pmaps[i,j,:,:])


    ppdf = pmaps / np.sum(pmaps,axis=(2,3),keepdims=True)
    mpdf = mmaps / np.sum(mmaps,axis=(2,3),keepdims=True)


    def KLdiv(p,q):
        ok = np.logical_and(p > 0, q > 0);
        return np.sum(p[ok]*np.log2(p[ok]/q[ok]))
    
    kld = np.zeros((nbins, nfm))
    for i in range(nbins):
        for j in range(nfm):
            kld[i,j] = KLdiv(mpdf[i,j,:,:],mpdf[0,j,:,:])
            

#        fmaps[(i*batch_size):(i*batch_size + batch_size),:] = FEATURE_MAPS
        
#    ii = np.argsort(cpred)
#    jj = np.argsort(cpath)
#    
#    plt.figure()
#    plt.plot(cpred[ii], fmaps[ii,:])
#    plt.title('model x')
#
#    plt.figure()
#    plt.plot(cpath[jj], fmaps[jj,:])
#    plt.title('pathologist x')

#   The following correlates very well with cpath. 
#        plt.plot(cpath[jj], fmaps[jj,165]/(fmaps[jj,430]+fmaps[jj,193]))
        

    