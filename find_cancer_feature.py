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

FEATURE_MAPS = 'nothing to see here yet'  # a global to hold maps obtained via hook functions. 


n_class = 1
GPU = True
batch_size = 20

dldb_path = \
'/media/bill/Windows1/Users/peria/Desktop/work/Brent Lab/'+\
'Boucheron CNNs/DLDBproject/DLDB_20181015_0552'

db = dldb.DLDB(input_directory = dldb_path)

#  This fcn is the one that I trained with a dropout layer in place. I then 
#  began to think that might be a bad idea, so I removed the dropout layer. 
#            fcn_name = '/media/bill/Windows1/Users/' + \
#                              'peria/Desktop/work/Brent Lab/Boucheron CNNs/'+\
#                              'DLDBproject/preFCN20181106_2245'

#
#   The following fcn is the one I got by removing the dropout layer and training      
#   the decoder again. This did not change the "zeroing feature maps has hardly
#   any effect" mystery.          
#            
#fcn_name = '/media/bill/Windows1/Users/' + \
#                  'peria/Desktop/work/Brent Lab/Boucheron CNNs/'+\
#                  'DLDBproject/preFCN20181128_1130'
#
#   The following fcn is the one I made after discovering that loading the state_dict from an FCN file
#   actually also overwrites the VGG coefficients. The fcn.py code still behaves that way, but now
#   I use the requires_grad  = False option to freeze the VGG part of the model, i.e. to freeze the "left-
#   half of the U" as Roger says it. 
#
#            
#fcn_name = '/media/bill/Windows1/Users/' + \
#                  'peria/Desktop/work/Brent Lab/Boucheron CNNs/'+\
#                  'DLDBproject/preFCN20190119_1603'

#
# Just use the "cancer detector" FCN, not the decoder. 
#
#fcn_name = '/media/bill/Windows1/Users/' + \
#                  'peria/Desktop/work/Brent Lab/Boucheron CNNs/'+\
#                  'DLDBproject/FCN20181205_2144'
#
# I think I have finally figured out that the VGG numbers I want are the ones found in 
#  preFCN20181128_1130. During the 1128 training session, these numbers were written over 
#  whatever VGG numbers I had thought I had froaen in the model. But these are the VGG 
#  numbers that include feature 165 as the "cancer feature". 
# So I need to load a cancer detector type model, with n_class=1, and then overwrite the 
#   pretrained part of the network with the state_dict of preFCN20181128_1130. 
#
#vggname = '/media/bill/Windows1/Users/'+\
#            'peria/Desktop/work/Brent Lab/Boucheron CNNs/'+\
#            'DLDBproject/vgg20181205_2144'
#vggname = None
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

#vggcopynet = fcn.load_model(n_class=3, load_encoder=False, fcnname='preFCN20181128_1130')
#
#new_dict = copy.deepcopy(net.state_dict())
#for k in vggcopynet.state_dict().keys():
#    if 'pretrained' in k:
#        print(k)
#        new_dict[k] = copy.deepcopy(vggcopynet.state_dict()[k])
#
#net.load_state_dict(new_dict)        
    
#torch.save(net.state_dict(),'redo_translate_for_165' + db.date_for_filename())

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
#        #
#        # This next expression is so pythonic it makes me want to pyvomit. All 
#        #   it does, I swear to God, is make it so I can broadcast im0 and denom. 
#        #
#        newshape = tuple([ns for ns in \
#                          itertools.chain.from_iterable(\
#                                            [[len(im0)],np.ones_like(axes)])])
#
#     OH!!! This is what the keepdims keyword is for. My bad.         
#
#        out = im - np.reshape(im0, newshape)
#
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
#    d = np.transpose(d.detach().cpu().numpy(),axes=(0,2,3,1))
#    d = (d-np.min(d))/(np.max(d)-np.min(d))
#    m = m.detach().cpu().numpy()

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

def lit_up_hook(self, input, output):
    global FEATURE_MAPS
    pred = output.cpu().detach().numpy()
    print(pred.shape)
#    p0 = np.min(np.min(np.min(pred, axis=-1, keepdims=True), axis=-1, keepdims=True), axis=-1, keepdims=True)
#    p1 = np.max(np.max(np.max(pred, axis=-1, keepdims=True), axis=-1, keepdims=True), axis=-1, keepdims=True)
#    pred = (pred-p0)/(p1-p0)
#    pred = pred/p1
    FEATURE_MAPS = np.mean(np.mean(np.abs(pred), axis=-1),axis=-1)
#    FEATURE_MAPS = np.mean(np.mean(pred - np.min(pred), axis=-1),axis=-1)
#    FEATURE_MAPS = np.std(np.std(pred, axis=-1),axis=-1)
#    FEATURE_MAPS = np.mean(np.mean(np.abs(pred), axis=-1),axis=-1)
#    FEATURE_MAPS = np.mean(np.mean(sigmoid(pred), axis=-1), axis=-1)
    
    print(FEATURE_MAPS.shape)
    
    nexamp, nfm, ny, nx = pred.shape
    FMsum = np.sum(pred, axis = (2,3))
    

    print('largest features are', np.argmax(FMsum,axis=1))

    return 

def show_ranges_hook(self, input, output):
    global FEATURE_MAPS
    pred = output.cpu().detach().numpy()
    
    for i in input:
        print(type(input))
        
    print('min is',np.min(pred),', max is',np.max(pred),', std is',np.std(pred))
    FEATURE_MAPS = np.std(pred,axis=(2,3))
    

def delentropy_hook(self, input, output):
    global FEATURE_MAPS
    pred = output.cpu().detach().numpy()
    
    
    nexamp, nfm, ny, nx = pred.shape
    pred = np.transpose(pred, axes=(0,2,3,1))
    pred = (pred-np.min(pred))/(np.max(pred)-np.min(pred))*128
    FMent = np.zeros((nexamp, nfm))
    for i in range(nexamp):
        FMent[i,:] = bu.delentropy(pred[i,:,:,:])
                
    FEATURE_MAPS = FMent
    
    print('largest features are', np.argmax(FMent,axis=1))

    return 


if __name__ == '__main__':
    
    modules = [module for name, module in net.named_modules()\
                     if len(module._modules) == 0]
    mnames = [name for name, module in net.named_modules()\
                     if len(module._modules) == 0]
    pick = 29 # this yields pretrained_net 28
    print('Examining feature maps of',mnames[pick])

    ntry = 10
    
    cpred = np.zeros(batch_size*ntry)
    cpath = np.zeros_like(cpred)
    fmaps = np.zeros((batch_size*ntry, 512))
    
    for i in range(ntry):
        print('Processing batch',i+1,'of',ntry)
        
        indata,y = grab_new_batch()
#        hook = modules[pick].register_forward_hook(lit_up_hook)
        hook = modules[pick].register_forward_hook(delentropy_hook)
#        hook = modules[pick].register_forward_hook(show_ranges_hook)
        out = torch.sigmoid(net(indata))
        hook.remove()

        cpath[(i*batch_size):(i*batch_size + batch_size)] = \
        np.mean(np.mean(y.cpu().detach().numpy(),axis=-1),axis=-1)
        
        cpred[(i*batch_size):(i*batch_size + batch_size)] = \
        np.mean(np.mean((out.cpu().detach().numpy() > 0.9),axis=-1),axis=-1)
        
        fmaps[(i*batch_size):(i*batch_size + batch_size),:] = FEATURE_MAPS
        
        
    ii = np.argsort(cpred)
    jj = np.argsort(cpath)
    
    plt.figure()
    plt.plot(cpred[ii], fmaps[ii,:])
    plt.title('model x')

    plt.figure()
    plt.plot(cpath[jj], fmaps[jj,:])
    plt.title('pathologist x')

#   The following correlates very well with cpath. 
#        plt.plot(cpath[jj], fmaps[jj,165]/(fmaps[jj,430]+fmaps[jj,193]))
        

    