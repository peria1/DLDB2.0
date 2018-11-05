#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 16:54:28 2018

@author: bill
"""
import openslide
import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
import platform as platf
import tkinter as tk
import pandas as pd

class WSIseg():
    def __init__(self, input_directory = None,  file_type = 'svs',RBMD = None,
                  sheet_name = None):
        file_type  = 'svs'
        if input_directory == None:
            input_directory = self.uichoosedir()

  #      print(input_directory,self.get_slash(),file_type)
        RBMD = '/media/bill/Windows1/Users/peria/Desktop/work/writingProjects/CellNet/NN Image metadata 20180710 WJP.xlsx'
        sheet_name='Merged Master Inventory'
        md = pd.read_excel(RBMD,sheet_name = sheet_name, 
                           dtype=str,converters = {'Magnification': int})
        tags = md.UniqueID
    
        newSourceImages = sorted(glob.glob(input_directory + self.get_slash() + '*.' + file_type))
        
        segments = {}
        filesDone = 0
        totalToDoStr = str(len(newSourceImages))
        for file in newSourceImages:
            this_row = md.loc[tags + '.' + file_type == self.just_filename(file)]
            print(self.just_filename(file))
            print(this_row["NumberOfSegments"].values[0] + ' segments...')
            
            ts = self.TissueSeg(file)
            while not ts.done:
                plt.pause(0.5)
            segments.update({self.just_filename(file): ts.polygons})
            filesDone += 1
            print('Finished ' + str(filesDone) + ' of ' + totalToDoStr)
            
        self.segments = segments
        
        output_file = input_directory + self.get_slash() + 'segments.dat'
#        with open(output_file, 'w') as file:
#             file.write(pickle.dumps(self.segments)) # use `pickle.loads` to do the reverse

        with open(output_file,'wb') as file:
            file.write(pickle.dumps(segments))
            
#        with open(output_file,'rb') as file:
#            stuff = pickle.loads(file.read())



    def uichoosedir(self): # not even kidding
        from tkinter.filedialog import FileDialog
        root = tk.Tk()
        root.focus_force()
        root.withdraw() # we don't want a full GUI, so keep the root window 
                        #  from appearing
        pathname = tk.filedialog.askdirectory()
        return pathname

    def just_filename(self, path):
        return path.split(sep=self.get_slash())[-1]

    def get_slash(self):
        if platf.system() == 'Windows':
            slash = '\\' # So pythonic!! Duplicit is better than complicit. 
        else:
            slash = '/'
        return slash


    class TissueSeg():
        
        def GPchoosefile(self):
            import tkinter as tk
            from tkinter.filedialog import FileDialog
            root = tk.Tk()
            root.withdraw() 
            filename = tk.filedialog.askopenfilename()
            return filename
 
        def onclick(self, event):
            if event.xdata == None: # click was outside axes, all done!
                self.done = True
                plt.close()
                
            if event.button == 3:   # indicate a polygon is finished with right-click
                self.xpoly.append(self.xpoly[0])
                self.ypoly.append(self.ypoly[0])
#                for x in self.xpoly:
#                    x *= self.scale
#                for y in self.ypoly:
#                    y *= self.scale
                
                plt.plot(self.xpoly,self.ypoly)
                self.fig.canvas.draw()
                this_poly = list(zip(np.rint(np.asarray(self.xpoly)*self.scale).astype(int),\
                                     np.rint(np.asarray(self.ypoly)*self.scale).astype(int)))
                #print(this_poly)
                self.polygons.append(this_poly)
                self.xpoly[:] = []
                self.ypoly[:] = []        
                return
            else:
                self.xpoly.append(event.xdata)
                self.ypoly.append(event.ydata) 
        
                if len(self.xpoly) > 1:
                    plt.plot(self.xpoly[-2:], self.ypoly[-2:])
                    self.fig.canvas.draw()
                else:
                    plt.plot(self.xpoly,self.ypoly,'o')
                    self.fig.canvas.draw()

        def __init__(self, imageFile = None):            
            plt.clf()
            if imageFile == None:
                imageFile = self.GPchoosefile()
                
            I = openslide.open_slide(imageFile)
            level_dims = I.level_dimensions 
            self.scale = I.level_downsamples[-1]
                                    
            J_whole = np.asarray(I.read_region((0,0),len(level_dims)-1,level_dims[-1]))   
            plt.imshow(J_whole)
            plt.show()
            
            self.fig = plt.gcf()
        
            self.xpoly = []
            self.ypoly = []
            
            self.polygons = []
            
            self.file = imageFile
            
            self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
            self.done = False

    