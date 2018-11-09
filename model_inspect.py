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

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import dldb
from dldb import dlTile


class Model:
    
    def __init__(self):
        #
        # Need to eventually work out of a model dictionary or some such. For
        #  now, boneheaded hard-coded paths will have to do. 
        #
        import fcn
        
        pth = '/media/bill/Windows1/Users/'+\
        'peria/Desktop/work/Brent Lab/Boucheron CNNs/'+\
        'DLDBproject/DLDB_20181015_0552'

        self.model = fcn.load_model(n_class=3,\
                              fcnname='/media/bill/Windows1/Users/' + \
                              'peria/Desktop/work/Brent Lab/Boucheron CNNs/'+\
                              'DLDBproject/preFCN20181106_2245')
        
        self.model.db = dldb.DLDB(pth)

        
        self.batch_size = 20
        self.GPU = True
        self.input = self.grab_new_batch()
        
    def get_model(self):
        return self.model
    
    def grab_new_batch(self,N=None, maskfile = None, augment = False, boundary_kernel=None):

        if N == None:
            N=list(np.random.randint(0,size=self.batch_size,high=1260))    

        indata = self.db.feed_pytorch(N=N)
        
        if self.GPU:
            indata = indata.cuda()
        
        return indata

    def display_feature(self,v):
        print(v.get())
        print(self.model.state_dict()[v.get()].size())
        
    def feature_hook(self, input, output):
        print(input.size)
        
#    def calculate(self):
#        x, y = np.meshgrid(np.linspace(-5, 5, self.xpoint), np.linspace(-5, 5, self.ypoint))
#        z = np.cos(x ** 2 * y ** 3)
#        self.res = {"x": x, "y": y, "z": z}



class View:
    def __init__(self, root, model):
        self.root = root   
        self.frame = Tk.Frame(root)
        
        self.model = model.get_model()
        
        self.fig = Figure(figsize=(7.5, 4), dpi=80)
        self.ax0 = self.fig.add_axes((0.05, .05, .90, .90), \
                                     facecolor=(.75, .75, .75), frameon=False)
        self.frame.pack(side=Tk.LEFT, fill=Tk.BOTH, expand=1)
 #------------------------       
        self.frame2 = Tk.Frame(root)
        self.frame2.pack(side=Tk.LEFT, fill=Tk.BOTH, expand=1)
 
        self.plotBut = Tk.Button(self.frame2, text="Plot ")
        self.plotBut.pack(side="top", fill=Tk.BOTH)
        self.plotBut.bind("<Button>", self.plot)

        optionList = tuple([key for key,item in self.model.named_parameters()])
        self.v = Tk.StringVar(master=self.frame2,name="feature")
        self.v.set(optionList[0])
        
        # The second argument in the next line is the name of the function that is 
        #   called when v is changed, i.e. when the user picks a new option from 
        #   paramMenu. It has to be a lambda (an anonymous function) to accomodate
        #   the additionsl arguments that I want to pass to my display_feature code. 
        # I think that *_ represents the set of arguments that trace_add passes in 
        # the background...I don't get to control that. 
        #
        self.v.trace_add("write",lambda *_, var=self.v : model.display_feature(var))
        
        self.paramMenu = Tk.OptionMenu(self.frame2, self.v, *optionList)
        self.paramMenu.pack(side="top",fill=Tk.BOTH)
                
        self.quitButton = Tk.Button(self.frame2, text="Quit")
        self.quitButton.pack(side="top", fill=Tk.BOTH)
        self.quitButton.bind("<Button>", self.quitit)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        self.canvas.draw()

    def clear(self, event):
        self.ax0.clear()
        self.fig.canvas.draw()

    def plot(self, event):
         tile = self.model.db.get_random_tile()
         self.ax0.clear()
         tile.show(self.ax0)
         self.fig.canvas.draw()

    def quitit(self,event):
        self.root.destroy()

        
class Controller:
    def __init__(self):
        self.root = Tk.Tk()
        self.model = Model()
        self.view = View(self.root, self.model)

    def run(self):
        self.root.title("Tkinter MVC example")
        self.root.deiconify()
        self.root.mainloop()

if __name__ == '__main__':
    c = Controller()
    c.run()

