#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:47:26 2018

@author: bill
"""


try:
    import Tkinter as Tk # python 2
except ModuleNotFoundError:
    import tkinter as Tk # python 3


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import time

import dldb
from dldb import dlTile

import torch

FEATURE_MAPS = 'nothing to see here yet'  # a global to hold maps obtained via hook functions. 

class Model:
    def __init__(self,batch_size=20,GPU=True):
        #
        # Need to eventually work out of a model dictionary or some such. For
        #  now, boneheaded hard-coded paths will have to do. 
        #
        import fcn
        
        self.batch_size = batch_size
        self.GPU = GPU
        
        self.icurrent = 0

        pth = '/media/bill/Windows1/Users/'+\
        'peria/Desktop/work/Brent Lab/Boucheron CNNs/'+\
        'DLDBproject/DLDB_20181015_0552'

        self.net = fcn.load_model(n_class=3,\
                              fcnname='/media/bill/Windows1/Users/' + \
                              'peria/Desktop/work/Brent Lab/Boucheron CNNs/'+\
                              'DLDBproject/preFCN20181106_2245').eval()
        if self.GPU:
            self.net.cuda()
        
        self.module_names = [name for name, module in self.net.named_modules()\
                             if len(module._modules) == 0]
        self.modules = [module for name, module in self.net.named_modules()\
                             if len(module._modules) == 0]
        
        self.db = dldb.DLDB(pth)

#        self.input, self.masks = self.grab_new_batch(N=[318, 376, 448, 819, 979],\
#                                         maskfile='test3_cancer.tif')

        self.get_new_data() # sets self.input and self.masks 
        
        self.selected_feature_maps = []
        self.viewers = []
        
    def update_viewers(self):
        for v in self.viewers:
            v.update_plots()

    def add_viewer(self,v):
        self.viewers.append(v)
        
    def get_new_data(self):
        Nlist = list(np.random.randint(0,high=1260,size=self.batch_size))
        self.input, self.masks = self.grab_new_batch(N=Nlist,\
                                         maskfile='test3_cancer.tif')
        self.icurrent = 0
        
    def next_example(self):
        self.icurrent = (self.icurrent + 1) % self.batch_size
    def prev_example(self):
        self.icurrent = (self.icurrent - 1) % self.batch_size
        
    def set_selected_feature_map_weights_to_zero(self, v):
        picked_module_name = v.get()
        pick = [i for i,n in enumerate(self.module_names)\
                if n == picked_module_name][0]

        print('MAD of weight is ',np.mean(np.abs(self.modules[pick].weight.cpu().detach().numpy())))
        print('MAD of bias is ',np.mean(np.abs(self.modules[pick].bias.cpu().detach().numpy())))

        print('Setting all maps to zero in ', picked_module_name)
        
        print(self.modules[pick])
        mp = self.modules[pick]
        print(mp)
        
        print(self.modules[pick].weight.size())
        
        param_names = [name for name,param in self.net.named_parameters()]
        ppick = [i for i,n in enumerate(param_names)\
                 if n == picked_module_name + '.weight'][0]
        
        params = torch.nn.ParameterList(param for param in self.net.parameters())
        param_picked = params[ppick]
        print(param_picked.size())
        
        
        pw = [param for name,param in self.net.named_parameters() \
              if name == picked_module_name + '.weight'][0]
        pb = [param for name,param in self.net.named_parameters() \
              if name == picked_module_name + '.bias'][0]
        
        print('training is: ',self.net.training)
        out0 = self.get_data_for_display(output=True)
        for fmap in self.selected_feature_maps:
#            self.modules[pick].weight[:,fmap,:,:]=0.0
#        for fmap in range(self.modules[pick].weight.size()[1]):
#            self.modules[pick].weight[:,fmap,:,:]=0.0
#            self.modules[pick].bias[fmap] = 0.0
            param_picked[fmap,:,:,:]=0.0
         #   param_picked.bias[fmap] = 0.0
        out1 = self.get_data_for_display(output=True)
        print('Change: ',np.sum(np.abs(out1-out0)))
            
#        self.modules[pick].weight[:,:,:,:]=0.0
#        self.modules[pick].bias[:] = 0.0
        print(np.sum(np.abs(pw.cpu().detach().numpy())))
        print(np.sum(np.abs(pb.cpu().detach().numpy())))
        print(np.sum(np.abs(mp.weight.cpu().detach().numpy())))
        print(np.sum(np.abs(mp.bias.cpu().detach().numpy())))
        
        fmap = self.selected_feature_maps[0]
        print(np.sum(np.abs(self.modules[pick].weight.cpu().detach().numpy()[fmap,:,:,:])))
        print(np.sum(np.abs(self.modules[pick].bias.cpu().detach().numpy()[fmap])))

#        handle = self.modules[pick[0]].register_forward_hook(zero_hook)
#        IDX = self.selected_feature_maps
        print(id(self.modules[pick]))
        print(id(mp))
        print(id(param_picked))
        
        self.net(self.input)
        self.update_viewers()
#        handle.remove()


    def get_data_for_display(self, output = False):
        if output:
            stuff = self.net(self.input)
        else:
            stuff = self.input
        
#        print(stuff.size())
        data = stuff.cpu().detach().numpy()
        if len(data.shape) < 4:
            data = np.expand_dims(data,0)
        
        dmin = np.min(data); dmax = np.max(data)
        data = (data - dmin)/(dmax-dmin)
        data = np.transpose(data[self.icurrent,:,:,:],axes=[1,2,0])
        return data
    
    def get_mask_for_display(self):
        m = self.masks[self.icurrent,0,:,:]
        return m
    
    def get_feature_map_for_display(self):
        global FEATURE_MAPS
        
        if type(FEATURE_MAPS) is not str:
            self.make_feature_map_display(FEATURE_MAPS) 
            fm = self.feature_display[self.icurrent,:,:]
            return fm
        else:
            return np.asarray([[1,0],[0,1]])
    
    def get_net(self):
        return self.net
    
    def grab_new_batch(self,N=None, maskfile = None, augment = False, boundary_kernel=None):

        if N == None:
            N=list(np.random.randint(0,size=self.batch_size,high=1260))    

        indata, y = self.db.feed_pytorch(N=N, maskfile=maskfile)
        
        if self.GPU:
            indata = indata.cuda()
            if y is not None:
                y = y.cuda()
        
        return indata, y

    def set_feature_map_hook(self,v):
        picked_module_name = v.get()
        
        pick = [i for i,n in enumerate(self.module_names)\
                if n == picked_module_name]
        print(self.module_names[pick[0]])
        
#        handle = self.modules[pick[0]].register_forward_hook(lambda x,y,z: print(y[0].size()))
#        handle = self.modules[pick[0]].register_forward_hook(feature_hook)
        handle = self.modules[pick[0]].register_forward_hook(capture_data_hook)
#        hthis = self.modules[pick[0]].register_forward_hook(summary_hook)
#        hnext = self.modules[pick[0]+1].register_forward_hook(summary_hook)
        self.net(self.input)
#        print('not removing hooks!!!')
        handle.remove()
#        hthis.remove()
#        hnext.remove()
        
#        print('Leaving hook set....')
    
    def make_feature_map_display(self, feat, point_clicked = None):
        
        nexamp,ndepth,nx,ny = feat.shape
        
        # find nearly square factors to cover depth, i.e. integer factors that bracket the 
        # square root as closely as possible.  
        #
        nrows, ncols = best_square(ndepth)
               
        assert(nx==ny)
        sqsize = nx
        brdr = nx // 8
        if brdr == 0:
            brdr = 1
            
        disp = np.zeros((nexamp,nrows*(sqsize+brdr),ncols*(sqsize+brdr)))
        
        def get_feature_corners(i):
            irow = i // ncols 
            icol = i %  ncols
            r0 = (sqsize+brdr)*irow
            r1 = r0 + sqsize
            c0 = (sqsize+brdr)*icol
            c1 = c0 + sqsize
            return r0,r1,c0,c1

        def highlight_border(i):
            r0,r1,c0,c1 = get_feature_corners(i)
            tall_top = r0
            tall_bottom = tall_top + sqsize + brdr
            tall_left = c1
            tall_right = tall_left + brdr
            
            fat_top = r1
            fat_bottom = fat_top + brdr
            fat_right = c1
            fat_left = fat_right - sqsize
#            print('setting border to zero for feature',i)
#            print('fat LRTB: ',fat_left,fat_right,fat_top,fat_bottom)
#            print('tall LRTB: ',tall_left,tall_right,tall_top,tall_bottom)
#            
#            print(disp[:, fat_top :fat_bottom,  fat_left :fat_right])
            disp[:, fat_top :fat_bottom,  fat_left :fat_right] = 0
            disp[:, tall_top:tall_bottom, tall_left:tall_right] = 0
#            print(disp[:, fat_top :fat_bottom,  fat_left :fat_right])

        print('about to initialize disp...')    
        for examp in range(nexamp):
            disp[examp,:,:] = np.max(feat[examp,:,:,:]) # per example, for max contrast
            for i in range(ndepth):
                r0,r1,c0,c1 = get_feature_corners(i)
                disp[examp,r0:r1,c0:c1] = feat[examp,i,:,:]
                
        if point_clicked is not None:
            px, py = point_clicked
            ifm1 = int(np.floor(py/(sqsize+brdr))*ncols)
            ifm2 = int(np.floor(px/(sqsize+brdr)))
            ifm = ifm1 + ifm2
            print('You clicked feature map number: ',ifm)
            if ifm not in self.selected_feature_maps:
                self.selected_feature_maps.append(ifm)
            else:
                self.selected_feature_maps.remove(ifm)
        
        for i in self.selected_feature_maps:
            highlight_border(i)
            
        self.feature_display = disp

def best_square(n):
    from sympy import factorint
    import itertools 
            
    # find nearly square factors to cover depth
    pfd = factorint(n)  # the prime factors of ndepth, which is the number of feature maps
    f = []
    for key,val in pfd.items():   # pfd is a dictionary of factors and number of times they 
        f.append([key]*val)       # occur, e.g. 24 -> {{2,3},{3,1}} . So expand it.... 
    f=list(itertools.chain.from_iterable(f))  #...and then flatten it. 
    f.sort()

    prod = 1
    pmax = int(n**0.5)
    for pf in f:
        if prod*pf <= pmax:
            prod *= pf
        else:
            nrows = prod
            ncols = n // nrows
            break
        
    return nrows, ncols


def capture_data_hook(self, input, output):
    global FEATURE_MAPS, IDX
    print('Hooked!')
    FEATURE_MAPS = output.cpu().detach().numpy()
        
def summary_hook(self, input, output):
    print('input size:',input[0].size())
    print('input sum:',np.sum(np.abs(input[0].cpu().detach().numpy())))
    print('output size:',output.size())
    print('output sum:',np.sum(np.abs(output.cpu().detach().numpy())))


#class FeatureExtractor(nn.Module):
#    def __init__(self, submodule, extracted_layers):
#        self.submodule = submodule
#
#    def forward(self, x):
#        outputs = []
#        for name, module in self.submodule._modules.items():
#            x = module(x)
#            if name in self.extracted_layers:
#                outputs += [x]
#        return outputs + [x]
#

class View:

    def __init__(self, root, model):
        self.root = root   
        self.frame = Tk.Frame(root)
        
        self.model = model
        self.net = model.get_net()
        
#        self.fig = Figure(figsize=(7.5, 4), dpi=80) # original
        self.fig = Figure(figsize=(16, 8), dpi=80)
        gs = gridspec.GridSpec(3,3)
        self.ax1 = self.fig.add_subplot(gs[0,0])
        self.ax2 = self.fig.add_subplot(gs[0,1])
        self.ax3 = self.fig.add_subplot(gs[0,2])
        self.ax4 = self.fig.add_subplot(gs[1:,0:])
        
#        axes = [self.ax1, self.ax2, self.ax3, self.ax4 ]
        
        
        
#        def process_button(event):
#            print("Button:", event.x, event.y, event.xdata, event.ydata, event.button)
#
##        fig.canvas.mpl_connect('key_press_event', process_key)
#        self.fig.canvas.mpl_connect('button_press_event', process_button)

#        self.ax0 = self.fig.subplots(3,3)
#        self.ax0 = self.fig.add_axes((0.05, .05, .90, .90), \
#                                     facecolor=(.75, .75, .75), frameon=False)
        self.frame.pack(side=Tk.LEFT, fill=Tk.BOTH, expand=1)

 #------------------------       
        self.frame2 = Tk.Frame(root)
        self.frame2.pack(side=Tk.LEFT, fill=Tk.BOTH, expand=1)
 
        self.plotBut = Tk.Button(self.frame2, text="Plot ")
        self.plotBut.pack(side="top", fill=Tk.BOTH)
        self.plotBut.bind("<Button>", self.plot)
#
#   Here are the essentials for setting arbitrary hooks based on a menu choice...
#
# mmod = [module for name, module in model.named_modules() if len(module._modules) == 0]
# handle = mmod[33].register_forward_hook(lambda x,y,z: print('Hello, world!'))
#
#   In the code below, the user has chosen to set a hook in module 33. mnames is
#        the list of module names. I want to use the menu choice from the optionList
#        widget to get the mname to get the module itself, to set the hook. 
#        
#    pick = [i for i,n in enumerate(mnames) if n is mnames[33]] 
#
#    handle.remove()
#    output = model(indata)
#    handle = mmod[pick[0]].register_forward_hook(lambda x,y,z: print('Hello, world!'))
#    output = model(indata)
#
#   These things should maybe be done inside the callback, where I will have good access
#        to the newest value of self.v, which is the name of the module in which 
#        to place the- hook. 

#        optionList = tuple([key for key,item in self.net.named_parameters()])
        optionList =[name for name, module in self.net.named_modules() \
                     if len(module._modules) == 0]
        self.v = Tk.StringVar(master=self.frame2,name="module")
        self.v.set('not set')
        #
#        def callback2(*_, var=self.v):
#            model.set_feature_map_hook(var)
#            crap = lambda *_ : self.plot()
#            crap(*_)
#            print('I can do two things..')
#            
#        def callback3(self, n,m,x, var=self.v):
#            model.set_feature_map_hook(var)
#            self.update_plots()
#            print('I could do three things..')
#
        # In the next line, "callback" is the name of the function that is 
        #   called when v is changed, i.e. when the user picks a new option from 
        #   paramMenu. It has to be a lambda (an anonymous function) to accomodate
        #   the additionsl arguments that I want to pass to my set_feature_map_hook code. 
        # I think that *_ represents the set of arguments that trace_add passes in 
        # the background...I don't get to control that. 
            
#        callback = lambda *_, var=self.v  : model.set_feature_map_hook(var)
        def module_callback(*_, var=self.v):
            model.set_feature_map_hook(var)
            model.selected_feature_maps = []
            self.update_plots()
        self.v.trace_add("write", module_callback)

        self.paramMenu = Tk.OptionMenu(self.frame2, self.v, *optionList)
        self.paramMenu.pack(side="top",fill=Tk.BOTH)
                
        self.nextButton = Tk.Button(self.frame2, text="Next")
        self.nextButton.pack(side="top", fill=Tk.BOTH)
        self.nextButton.bind("<Button>", self.next)
        self.prevButton = Tk.Button(self.frame2, text="Prev")
        self.prevButton.pack(side="top", fill=Tk.BOTH)
        self.prevButton.bind("<Button>", self.prev)

        
        self.zeroButton = Tk.Button(self.frame2, text="Zero")
        self.zeroButton.pack(side="top", fill = Tk.BOTH)
        self.zeroButton.bind("<Button>", self.zero_callback)

        self.grabButton = Tk.Button(self.frame2, text="New Data")
        self.grabButton.pack(side="top", fill=Tk.BOTH)
        self.grabButton.bind("<Button>", self.grab)

        self.quitButton = Tk.Button(self.frame2, text="Quit")
        self.quitButton.pack(side="top", fill=Tk.BOTH)
        self.quitButton.bind("<Button>", self.quitit)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        self.canvas.draw()
                
        def process_button(event):
#            print('processing button...')
            point = (event.x, event.y)
#            for ax in axes:
#                print(ax.contains_point(point))

            datapoint = (event.xdata, event.ydata)
            if self.ax4.contains_point(point):
#                print('...right axes...')
                if type(FEATURE_MAPS) is not str:
#                    print('...FEATURE_MAPS are ready...')
                    model.make_feature_map_display(FEATURE_MAPS, point_clicked = datapoint)
                    model.update_viewers()

        self.fig.canvas.mpl_connect('button_press_event', process_button)


        self.update_plots()

    def next(self, event):
        self.model.next_example()
        self.plot(event)

    def prev(self, event):
        self.model.prev_example()
        self.plot(event)
            
    def zero_callback(self,event):
        self.model.set_selected_feature_map_weights_to_zero(self.v)

    def grab(self, event):
        self.model.get_new_data()
        self.update_plots()
    
    def clear(self, event):
        self.ax0.clear()
        self.fig.canvas.draw()

    def plot(self, event):
        self.update_plots()
        
    def update_plots(self):
#        tile = self.model.db.get_random_tile()
        self.ax1.clear() # inexplicably began causing trouble....Oh! matplotlib was only ever imported in my hook, which is global 
#        tile.show(self.ax0)
        self.ax1.imshow(self.model.get_data_for_display())
        self.ax2.imshow(self.model.get_data_for_display(output=True))
        self.ax3.imshow(self.model.get_mask_for_display())
        img_fm = self.model.get_feature_map_for_display()
        self.ax4.imshow(img_fm)
        self.ax4.set_title(self.v.get())
        self.ax1.set_title(str(self.model.icurrent))
#        self.fig.colorbar(img_fm)
        self.fig.canvas.draw()
        
    def quitit(self,event):
        self.root.destroy()

    
class Controller:
    def __init__(self):
        self.root = Tk.Tk()
        self.model = Model()
        self.view = View(self.root, self.model)
        self.model.add_viewer(self.view)

    def run(self):
        self.root.title("Model Inspector")
        self.root.deiconify()
        self.root.mainloop()

if __name__ == '__main__':
    c = Controller()
    c.run()

