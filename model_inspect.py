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
import dldb
from dldb import dlTile

FEATURE_MAPS = 'nothing to see here yet'  # a global to hold maps obtained via hook functions. 

class Model:
    def __init__(self):
        #
        # Need to eventually work out of a model dictionary or some such. For
        #  now, boneheaded hard-coded paths will have to do. 
        #
        import fcn
        
        self.batch_size = 20
        self.GPU = True
        
        self.icurrent = 0

        pth = '/media/bill/Windows1/Users/'+\
        'peria/Desktop/work/Brent Lab/Boucheron CNNs/'+\
        'DLDBproject/DLDB_20181015_0552'

        self.model = fcn.load_model(n_class=3,\
                              fcnname='/media/bill/Windows1/Users/' + \
                              'peria/Desktop/work/Brent Lab/Boucheron CNNs/'+\
                              'DLDBproject/preFCN20181106_2245')
        if self.GPU:
            self.model.cuda()
        
        self.module_names = [name for name, module in self.model.named_modules()\
                             if len(module._modules) == 0]
        self.modules = [module for name, module in self.model.named_modules()\
                             if len(module._modules) == 0]
        
        self.db = dldb.DLDB(pth)

#        self.input, self.masks = self.grab_new_batch(N=[318, 376, 448, 819, 979],\
#                                         maskfile='test3_cancer.tif')

        self.get_new_data()

        
    def get_new_data(self):
        Nlist = list(np.random.randint(0,high=1260,size=self.batch_size))
        self.input, self.masks = self.grab_new_batch(N=Nlist,\
                                         maskfile='test3_cancer.tif')
        self.icurrent = 0
        
    def next_example(self):
        self.icurrent = (self.icurrent + 1) % self.batch_size
    def prev_example(self):
        self.icurrent = (self.icurrent - 1) % self.batch_size

    def data_for_display(self, output = False):
        if output:
            stuff = self.model(self.input)
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
    
    def mask_for_display(self):
        m = self.masks[self.icurrent,0,:,:]
        return m
    
    def feature_map_for_display(self):
        global FEATURE_MAPS
        
        if type(FEATURE_MAPS) is not str:
            self.make_feature_map_display(FEATURE_MAPS) 
            fm = self.feature_display[self.icurrent,:,:]
            return fm
        else:
            return np.asarray([[1,0],[0,1]])
    
    def get_model(self):
        return self.model
    
    def grab_new_batch(self,N=None, maskfile = None, augment = False, boundary_kernel=None):

        if N == None:
            N=list(np.random.randint(0,size=self.batch_size,high=1260))    

        indata, y = self.db.feed_pytorch(N=N, maskfile=maskfile)
        
        if self.GPU:
            indata = indata.cuda()
            if y is not None:
                y = y.cuda()
        
        return indata, y

    def display_feature(self,v):
        picked_module_name = v.get()
        
        pick = [i for i,n in enumerate(self.module_names)\
                if n == picked_module_name]
        print(self.module_names[pick[0]])
        
#        handle = self.modules[pick[0]].register_forward_hook(lambda x,y,z: print(y[0].size()))
#        handle = self.modules[pick[0]].register_forward_hook(feature_hook)
        handle = self.modules[pick[0]].register_forward_hook(capture_data)
        self.model(self.input)
        handle.remove()
#        print('Leaving hook set....')
    
    def make_feature_map_display(self, feat):
        
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
        
        for examp in range(nexamp):
            disp[examp,:,:] = np.max(feat[examp,:,:,:])
            for i in range(ndepth):
                irow = i // ncols 
                icol = i %  ncols
                r0 = (sqsize+brdr)*irow
                r1 = r0 + sqsize
                c0 = (sqsize+brdr)*icol
                c1 = c0 + sqsize
                disp[examp,r0:r1,c0:c1] = feat[examp,i,:,:]
                
        self.feature_display = disp
    
#------------------------
def register_nan_checks(model):
    def check_grad(module, grad_input, grad_output):
        # print(module) you can add this to see that the hook is called
        if any(np.all(np.isnan(gi.data.numpy())) for gi in grad_input if gi is not None):
            print('NaN gradient in ' + type(module).__name__)
    model.apply(lambda module: module.register_backward_hook(check_grad))
#------------------------


def capture_data(self, input, output):
    global FEATURE_MAPS, IDX
    print('Hooked!')
    FEATURE_MAPS = output.cpu().detach().numpy()

# DON'T EDIT THIS ONE YET!
def feature_hook(self, input, output):
    import matplotlib.pyplot as plt
#    plt.ion()
    
#    print('Inside hook, examining self...')
#    print(' ')
#    print('type: ',type(self))
#    print(' ')
#    print('dir: ',dir(self))
#    print(' ')
#    print('_get_name: ',self._get_name())
#    print(' ')
#    print('named_modules: ', [m for m in self.named_modules()])
#    print(' ')
#    print('__repr__: ',self.__repr__())
#    print(' ')
#    print('extra_repr: ',self.extra_repr())
#    print(' ')
##    print('named_parameters: ',[c for c in self.named_parameters()])    
#    print('type input: ',type(input))
#
##    input[0][:,87,:,:] = 1.0

#    print('output size: ', output.size())
    print('HEY! I am zeroing feature map 222!!')
#    input[0][:,222,:,:] = 0.0
    output[:,222,:,:]=0
    
    print(output.size())
    
    feat = input[0].cpu().detach().numpy()
    nexamp,ndepth,nx,ny = feat.shape
    
    
    # find nearly square factors to cover depth, i.e. integer factors that bracket the 
    # square root as closely as possible.  
    #
    nrows, ncols = best_square(ndepth)
           
    assert(nx==ny)
    sqsize = nx
    brdr = 1
    disp = np.max(feat) + np.zeros((nexamp,nrows*(sqsize+brdr),ncols*(sqsize+brdr)))
    A = np.zeros((nexamp,ndepth,sqsize,sqsize))
    
    fig = plt.figure()  # a new figure window
    for examp in range(nexamp):
        for i in range(ndepth):
            irow = i // ncols 
            icol = i %  ncols
            r0 = (sqsize+brdr)*irow
            r1 = r0 + sqsize
            c0 = (sqsize+brdr)*icol
            c1 = c0 + sqsize
            disp[examp,r0:r1,c0:c1] = feat[examp,i,:,:]
            drange = np.max(disp[examp,:,:])-np.min(disp[examp,:,:])
            A[examp,i,:] = feat[examp,i,:,:]
            
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.imshow((disp[examp,:,:]-np.min(disp[examp,:,:]))/drange)
        fig.show()
        plt.pause(.5)

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
        self.net = model.get_model()
        
#        self.fig = Figure(figsize=(7.5, 4), dpi=80) # original
        self.fig = Figure(figsize=(16, 8), dpi=80)
        self.ax0 = self.fig.subplots(2,2)
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

#        optionList = tuple([key for key,item in self.model.named_parameters()])
        optionList =[name for name, module in self.net.named_modules() \
                     if len(module._modules) == 0]
        self.v = Tk.StringVar(master=self.frame2,name="module")
        self.v.set(optionList[0])
        #
#        def callback2(*_, var=self.v):
#            model.display_feature(var)
#            crap = lambda *_ : self.plot()
#            crap(*_)
#            print('I can do two things..')
#            
#        def callback3(self, n,m,x, var=self.v):
#            model.display_feature(var)
#            self.update_plots()
#            print('I could do three things..')
#
        # In the next line, "callback" is the name of the function that is 
        #   called when v is changed, i.e. when the user picks a new option from 
        #   paramMenu. It has to be a lambda (an anonymous function) to accomodate
        #   the additionsl arguments that I want to pass to my display_feature code. 
        # I think that *_ represents the set of arguments that trace_add passes in 
        # the background...I don't get to control that. 
            
#        callback = lambda *_, var=self.v  : model.display_feature(var)
        def callback(*_, var=self.v):
            model.display_feature(var)
            self.update_plots()
        self.v.trace_add("write", callback)

        self.paramMenu = Tk.OptionMenu(self.frame2, self.v, *optionList)
        self.paramMenu.pack(side="top",fill=Tk.BOTH)
                
        self.nextButton = Tk.Button(self.frame2, text="Next")
        self.nextButton.pack(side="top", fill=Tk.BOTH)
        self.nextButton.bind("<Button>", self.next)
        self.prevButton = Tk.Button(self.frame2, text="Prev")
        self.prevButton.pack(side="top", fill=Tk.BOTH)
        self.prevButton.bind("<Button>", self.prev)

        self.grabButton = Tk.Button(self.frame2, text="New Data")
        self.grabButton.pack(side="top", fill=Tk.BOTH)
        self.grabButton.bind("<Button>", self.grab)

        self.quitButton = Tk.Button(self.frame2, text="Quit")
        self.quitButton.pack(side="top", fill=Tk.BOTH)
        self.quitButton.bind("<Button>", self.quitit)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        self.canvas.draw()
        
        self.update_plots()

    def next(self, event):
        self.model.next_example()
        self.plot(event)

    def prev(self, event):
        self.model.prev_example()
        self.plot(event)
    
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
        self.ax0[0,0].clear() # inexplicably began causing trouble....Oh! matplotlib was only ever imported in my hook, which is global 
#        tile.show(self.ax0)
        self.ax0[0,0].imshow(self.model.data_for_display())
        self.ax0[0,1].imshow(self.model.data_for_display(output=True))
        self.ax0[1,0].imshow(self.model.mask_for_display())
        img_fm = self.model.feature_map_for_display()
        self.ax0[1,1].imshow(img_fm)
        self.ax0[1,1].set_title(self.v.get())
        self.ax0[0,0].set_title(str(self.model.icurrent))
#        self.fig.colorbar(img_fm)
        self.fig.canvas.draw()
        
    def quitit(self,event):
        self.root.destroy()

    
class Controller:
    def __init__(self):
        self.root = Tk.Tk()
        self.model = Model()
        self.view = View(self.root, self.model)

    def run(self):
        self.root.title("Model Inspector")
        self.root.deiconify()
        self.root.mainloop()

if __name__ == '__main__':
    c = Controller()
    c.run()

