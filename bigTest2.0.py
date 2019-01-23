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
#import torch.nn as nn
#import torch.optim as optim
#from torchvision import models
#from torchvision.models.vgg import VGG
import numpy as np
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy as sp
import imageio
import glob

import billUtils as bu
#from matplotlib.backends.backend_pdf import PdfPages
import sys
import dldb
from dldb import dlTile

MAX_WHITE = 0.9

"""
    Given a color micrograph of some Hematoxylin and Eosin stained tissue in 
    H_and_E, this function computes the mean value in  regions containing tissue, 
    for each color layer.
    
    The criterion "white" is just an empirical test for the absence of tissue, i.e. 
    the background is white. I don't test for proper H & E colors. So far it has
    seemed pretty safe. 
    
    Because I am only summing over some pixels, I think I need to loop over the 
    layers to get their mean values, i.e. I can't somehow "vectorize" this. 
    
"""
def get_tissue_normalization(H_and_E):
    nx, ny, nz = H_and_E.shape
    T0 = np.zeros((nz,))
    dT = np.zeros_like(T0)

    flatten = lambda A : np.reshape(A,(nx*ny,1)).transpose(1,0)
    
    white = np.min(H_and_E,axis=2) > (.92*255)
    white_frac = np.sum(white)/np.prod(H_and_E.shape[0:2])
    print('White fraction is {:5.3f}'.format(white_frac))

    if white_frac > MAX_WHITE:
        return T0, dT, white_frac
    
    must_be_tissue = flatten(white == 0) # must be tissue since it's not background

    for i in range(nz):
        T0[i] = (flatten(H_and_E[:,:,i])[must_be_tissue]).mean(-1)
        dT[i] = (flatten(H_and_E[:,:,i])[must_be_tissue]).std(-1)
        
    return T0, dT, white_frac

if __name__ == "__main__":

    openslide.Image.MAX_IMAGE_PIXELS = None # prevents DecompressionBomb Error

    batch_size, n_class, h, w = 20, 1, 256, 256

    GPU = True
    pretrained = False  # don't use the ImageNet pretraining...
    show_plots = 'plot' in sys.argv
#    show_plots = True
    
#    FCNdir = '/media/bill/Windows1/Users/peria/Desktop/work/Brent Lab/' + \
#    'Boucheron CNNs/DLDBproject/'
    FCNdir = './'
    
#    vggname = FCNdir + 'vgg20181024_0253'  # first useful cancer detector
#    fcnname = FCNdir + 'FCN20181024_0253' 
    vggname = FCNdir + 'vgg20181205_2144'  # cancer detector with per color normalization
    fcnname = FCNdir + 'FCN20181205_2144'
         
    m = Model(batch_size=batch_size, GPU=GPU, fcn_name=fcnname, vggname=vggname,\
              n_class=n_class)
    fcn_model = m.net.cuda().eval()
    for p in fcn_model.parameters():
        p.requires_grad = False
    
    instructions = 'Choose tissue folder...don''t click Ok until window looks blank,,,'
    dir_to_process = bu.uichoosedir(title=instructions) + '/*.tif'
    files_to_process = sorted(glob.glob(dir_to_process))
    files_to_process = [f for f in files_to_process if 'cpd' not in f]
    
    maskfile = None
#    files_to_process = ['/media/bill/Windows1/Users/peria/Desktop/work/Brent Lab/' + \
#    'Boucheron CNNs/NickReder/test4_annotated.tif']
#    maskfile = '../NickReder/test4_cancer.tif'

    for file_to_process in files_to_process:
    
        fileparts = file_to_process.split(sep='.')
        fileparts[-2] += '_cpd'
        output_file = '.'.join(fileparts)
            
        Tissue = openslide.open_slide(file_to_process)
        
        allgs = None
        if maskfile:
            gs =  openslide.open_slide(maskfile)
            allgs = np.asarray(gs.read_region((0,0),0,gs.dimensions))[:,:,0]
            allgs = 1.0 - allgs/np.max(allgs)
    
        allT = np.asarray(Tissue.read_region((0,0),0,Tissue.dimensions))[:,:,0:3]

        T0, dT, white_frac = get_tissue_normalization(allT)
        
        if white_frac > MAX_WHITE:
            print('All white, no tissue..., skipping', file_to_process)
        else:
            T0 = np.float32(T0)
            dT = np.float32(dT)  # for use in pytorch
            test_out = np.zeros_like(allT,dtype=np.float64)[:,:,0]
            
            nx, ny = 4096, 256  # nx and ny need to be powers of two. 4096 x 256 maxes out CUDA RAM
            NX, NY = Tissue.dimensions
            IX, IY = (0,0)  # This is where the current tile starts, i.e. its upper-left
            border = 64 # This many pixels wide will be discarded around the edge of
            b = border            #   each tile
            xmove = nx - 2*b
            ymove = ny - 2*b
            
            print('processing ',file_to_process,'...')
            with torch.no_grad():
                while IY + h < NY:
                    IX = 0
                    h = ny 
#                    while (IY + h > NY): 
#                        h//=2 
                    while IX +w < NX:
                        w = nx
#                        while IX + w > NX:
#                            w//=2
                        chnk = np.asarray(Tissue.read_region((IX,IY),0,(w,h)),\
                                                dtype=np.float32)[:,:,0:3]
                        chnk = (chnk - T0)/dT # T0 and dT have 3 elements, so they broadcast
                        chnk = np.transpose(chnk,axes=[2,0,1])
                        chnkfeed = torch.tensor(chnk).unsqueeze(0).cuda()
                        outchnk = \
                        torch.sigmoid(fcn_model(chnkfeed)).cpu().detach()
#                        test_out[IY:(IY+h),IX:(IX+w)] = outchnk
                        test_out[(IY+b):(IY+ny-b),(IX+b):(IX+nx-b)] = \
                        outchnk[b:(ny-b), b:(nx-b)]
                        
                        IX += xmove
                    IY += ymove
                    print(IY if IY < NY else NY,'of',NY)
                          
            fakegs = np.zeros_like(test_out, dtype=np.uint8)
            
            fakegs = np.uint8(255 * test_out)
        
            imageio.imwrite(output_file, fakegs)
     
            if show_plots:
                ncols = 1
                if maskfile:
                    nrows = 3
                    ifake = 2
                    itrue = 3
                else:
                    nrows = 2
                    ifake = 2
                    itrue = None
                    
                fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
                ax[0].imshow(allT)
                ax[1].imshow(fakegs)
                ax[1].set_title(' '.join([bu.just_filename(bu,fcnname),'with',bu.just_filename(bu,file_to_process)]))
                if allgs is not None:
                    ax[2].imshow(allgs)
                    ax[2].set_title(bu.just_filename(bu,maskfile))
    

#  I had some problems with normalization. Turns out it's a numpy bug.
#                
#    T0 = np.mean(allT,axis=(0,1),dtype=np.float32) # this yields some kind of roundoff error
#    dT = np.std(allT,axis=(0,1),dtype=np.float32)  # all three values are the same! 
#
#    T0 = np.float32(np.mean(allT,axis=(0,1))) # this gets an answer the appears correct
#    dT = np.float32(np.std(allT,axis=(0,1)))
#
#   I read some stuff about a robust pairwise summation algorithm that numpy uses, 
#   but apparently it only applies that over trailing dimensions. So I will transpose and
#   explicitly sum over each trailing dimension in turn. And doing them together does not
#   work! Look at this example:
#   np.float16(allT.transpose((2,0,1))).mean(axis=-1).mean(axis=-1)
#   
#   Out[108]: array([225.5, 202.1, 221.1], dtype=float16)
#   
#   Looks good! Now try to do them in one fell swoop...
#
#   np.float16(allT.transpose((2,0,1))).mean(axis=(-1,-2))
#
#   Out[110]: array([29.66, 29.66, 29.66], dtype=float16)
#    
#   Total nonsense!!! An error passing silently! By the way, all three values are literally
#     equal, not just numerically really close. Where do they come from? Just for fun, I
#    tried adding some constants into the averaging:
#
#np.float16(allT.transpose((2,0,1))+1.0).mean(axis=(-2,-1))
#Out[113]: array([29.66, 29.66, 29.66], dtype=float16)
#
#    HA! 1 == 0!! QED!!!
#    
#np.float16(allT.transpose((2,0,1))+1.0e6).mean(axis=(-2,-1))
#Out[114]: array([inf, inf, inf], dtype=float16)
#
#   See! A million IS a big number!!
#    
#np.float16(allT.transpose((2,0,1))+1.0e1).mean(axis=(-2,-1))
#Out[115]: array([59.3, 59.3, 59.3], dtype=float16)
#
#   And 10 is pretty close to 29.7! Is that roundoff error? 
#    
#np.float16(allT.transpose((2,0,1))+.5e1).mean(axis=(-2,-1))
#Out[116]: array([59.3, 59.3, 59.3], dtype=float16)#    
#    
# I would have a better attitude if I had been subjected to "The Zen of Python" less often.
#
