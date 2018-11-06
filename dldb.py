# -*- coding: utf-8 -*-
"""
dldb.py

Defines a class to represent an LMDB database containing image data of 
appropriate dimensions, and all necessary metadata, for deep learning.


Created on Wed Apr 25 05:52:46 2018

@author: Bill Peria
"""
import openslide
import numpy as np
#import scipy as sp
import skimage.transform
from skimage.transform import warp
import glob
import lmdb
import pickle
import os
#import sys
import matplotlib.pyplot as plt
import platform as platf
import time
import imageio
import pandas as pd
import tkinter as tk
#import gc
import torch

##
## It proved impossible to make a single LMDB that was large enough to hold 
##  the data sets that we want to use. 
#class Dlist():
#    def __init(self):
#        pass
# 
#  Below is a list of the methods currently in DLDB. I need to make sure that when 
#  I split into multiple LMDBs, all of these still work properly. 
#  
#  I also think there needs to be a master LMDB that contains the names and 
#  locations of all the other LMDBs, indexed by sourcefile. So all the tiles
#  from a particular #  slide go to a particular LMDB, but which one? 
#
# 'env',
# 'feed_caffe',
# 'get_metadata',
# 'get_minibatch',
# 'get_random_tile',
# 'get_slash',
# 'get_sourcefiles',
# 'get_tile_by_number',
# 'just_filename',
# 'makeTiles',
# 'record_metadata',
# 'stat',
# 'tile_has_tissue',
# 'time_of_day',
# 'uichoosedir']




class DLDB():
    def __init__(self, input_directory = None, visualize = False,
                  append_to=None, supervised=False,
                  tileSize=[299,299],use_level=2,
                  build = False, caffe_name = None, test_frac = 0.17,
                  use_metadata = False, file_type = 'svs',RBMD = None,
                  sheet_name = None, output_directory = None):
                
        metadata_loaded = False
        
        if RBMD == None:
            RBMD = '/media/bill/Windows/Users/peria/Desktop/work/writingProjects/CellNet/NN Image metadata 20180710 WJP.xlsx'
            sheet_name='Merged Master Inventory'
        
        ffs = use_level

        if not caffe_name == None:
            try:
                os.mkdir(caffe_name)
            except:
                print('Unable to create ' + caffe_name +'. Does it exist already?')
                return None
                    
        if build:
            map_size = 1e12 # does this reserve a terabyte of disk? 
           
            if input_directory==None:
                input_directory =self.uichoosedir()
            
            if output_directory == None:
                output_directory = input_directory
            
            dbname = output_directory + self.get_slash() + 'DLDB_' + self.date_for_filename()

            dx,dy = tileSize 
            
            #
            #  Each of our directories needs to contain a file listing of 
            #   polygons enclosing tissue segments; the file needs to be called segments.dat
            #
            #  The vertices are obtained by mouse clicks recorded by WSIseg().
            #
            #  If there is no segments.dat file, I just generate a polygon: a 
            #    rectangle around the whole slide. 
            #
            try:
                with open(input_directory + self.get_slash() + 
                          'segments.dat','rb') as file:
                    segs = pickle.loads(file.read())
                    no_segs = False
            except:
                no_segs = True
                        
            dir_to_process = input_directory + self.get_slash() + '*.' + file_type
            newSourceImages = sorted(glob.glob(dir_to_process))
           
#            pick = 0
#            print(type(newSourceImages[0]))
#            print('Just doing one file:' + newSourceImages[pick])
#            newSourceImages = [newSourceImages[pick]]
#            print(newSourceImages)
            
        
            env0 = lmdb.open(dbname, map_size = map_size, writemap=True)
            txn0 =  env0.begin(write=True)
            tileCount = 0
            maxTiles = 5
            tileIndex = {}
            nLMDB = 0
            LMDBlist = []
            # need to record header here. Creation date, path to files, number of files, metadata header, etc.
            txn0.put('number_of_source_files'.encode(),pickle.dumps(len(newSourceImages)))
            processedFiles = {}
            env1 = None
            txn1 = None
            newLMDB = None
            firstThisLMDB = 0
            print(newSourceImages)
            
            for currentFile in newSourceImages:
                tiles_to_start = tileCount

                if tileCount >= nLMDB * maxTiles:
                    if env1 != None:
                        tileIndex.update({newLMDB:(firstThisLMDB,tileCount)})
                        txn1.commit()
                        env1.close()

                    firstThisLMDB = tileCount
                    nLMDB = nLMDB + 1
                    newLMDB = dbname + '_' + str(nLMDB)
                    print('creating ' + newLMDB)
                    LMDBlist.append(newLMDB)
                        
                    env1 = lmdb.open(newLMDB, map_size = map_size, writemap = True)
                    txn1 = env1.begin(write=True)
                    
                print('Processing ' + self.just_filename(currentFile))
                try:
                    I = openslide.open_slide(currentFile) 
                    
                    if not no_segs:
                        polygons = segs[self.just_filename(currentFile)]
#                        print(type(polygons[0]))
                    else: # no segments or regions, just tile the whole thing
                        (x1,y1) = I.level_dimensions[0]
                        (x0,y0) = (0,0)
                        # A list of polygons is a list of lists of vertices
                        polygons = [[[x0,y0],[x1,y0],[x1,y1],[x0,y1],[x0,y0]]]
                    
                    tileCount = self.makeTiles(I, txn1, env0, dx, dy, currentFile, polygons, tileCount,\
                                   use_level = ffs, visualize = visualize)
                    print(tileCount)
                    
                    print('Made tiles ' + str(tiles_to_start) + '-' +  str(tileCount))
                    if tileCount > tiles_to_start:
                        if use_metadata:
                            if metadata_loaded == False:
                                md = pd.read_excel(RBMD,sheet_name = sheet_name, 
                                                   dtype=str,converters = {'Magnification': int})
                                metadata_loaded = True
                                tags = md.UniqueID
                        
                            this_row = md.loc[tags + '.' + file_type == self.just_filename(currentFile)]
                            stuff = {"number_of_tiles":tileCount}
                            if not this_row.empty:
                                for col in this_row.columns:
                                    stuff.update({col:this_row[col].values[0]})
                        
                            stuff.update(dict(I.properties)) # just slam the whole Aperio dictionary in there
                        
                            self.record_metadata(txn0,currentFile,stuff)
                            
                        processedFiles.update({self.just_filename(currentFile):[tiles_to_start,tileCount]})
                        print(self.just_filename(currentFile))
                        print(processedFiles)
                        
                except BaseException as e:
                    print(self.just_filename(currentFile) + ' failed...')
                    print(e)
                
            txn0.put('number_of_tiles'.encode(),pickle.dumps(tileCount))  # TODO
            txn0.put('processed files'.encode(),pickle.dumps(processedFiles))
            tileIndex.update({newLMDB:(firstThisLMDB,tileCount)})
            txn0.put('tile index'.encode(), pickle.dumps(tileIndex))
            txn0.put('input directory'.encode(), pickle.dumps(input_directory))
            
            
            mdl = []
            for fl in processedFiles.keys():
                byte_key = fl.encode()
                pickled_crap = txn0.get(byte_key)
                if not pickled_crap == None:
                    newdict = pickle.loads(pickled_crap)
                    mdl.append(newdict)

            if mdl:
                md = pd.DataFrame.from_dict(data=mdl)
                writer = pd.ExcelWriter('check_metadata.xlsx')
                md.to_excel(writer,'Sheet1')
                print('metadata are type ' + str(type(md)))
                txn0.put('metadata'.encode(),pickle.dumps(md))
                        
#            mdchk = pickle.loads(txn0.get('metadata'.encode()))
#            print('mdchk is ' + str(type(mdchk)))
            
            txn0.commit()
            txn1.commit()
            env0.close()
            env1.close()
            
            dbdir = dbname
            self.input_directory = input_directory
        else:
            if input_directory == None:
                dbdir = self.uichoosedir()
            else:
                dbdir = input_directory
                
        global envr
        envr = lmdb.open(dbdir)
        self.env = envr
        self.input_directory = self.lmdbget('input directory')

#        if not caffe_name == None:
#            print(caffe_name)
#            self.feed_caffe(caffe_name, test_frac)  
            
    def get_metadata(self):
        with self.env.begin(write=False) as txn:
            metadata = pickle.loads(txn.get('metadata'.encode()))
        return metadata
        
    def makeTiles(self, I, txn, env, dx, dy, currentFile, polygons, tileCount,\
                  use_level=None, visualize=False):
#
#   This function writes to an LMDB (via txn), all the instances of 
#   tissue-containing tiles of size (dx,dy) represented in slide I at level 
#   use_level. It returns a count of the  global number of tiles written 
#   so far in this call to build a DLDB. 
#
#        
        TISSUE_BOX = 'g-'
        EMPTY_BOX = 'r-'
        tileCountZero = tileCount
        
        ul_corners = []
        ultry_corners = []

        xsq = np.array([0,1,1,0,0])
        ysq = np.array([0,0,1,1,0])
        
        level_dims = I.level_dimensions 
        level_downs = I.level_downsamples 

        maxlev = len(level_downs)-1
        if use_level > maxlev:
            print('fixing use_level: ' +str(use_level) + ' -> '
                  + str(maxlev))
            use_level = maxlev
            
        ymax = int(level_dims[use_level][1]*level_downs[use_level])
        xmax = int(level_dims[use_level][0]*level_downs[use_level])
        dxscoot = int(dx * level_downs[use_level])
        dyscoot = int(dy * level_downs[use_level])
        J_whole = np.asarray(I.read_region((0,0),len(level_dims)-1,level_dims[-1]))   

        for y0 in range(0,ymax-dyscoot,dyscoot):
            for x0 in range(0,xmax-dxscoot,dxscoot):
                for sn,pg in enumerate(polygons):
                    if in_poly((x0+int(dx/2),y0+int(dy/2)),pg):
                        ultry_corners.append((x0,y0,sn))
                
        block = int(len(ultry_corners)/10)
        if block == 0:
            block = 1
        for corner in ultry_corners:
#
#  There used to be a call here to skimage.transform.resize. If we go back to that
#  e.g. to get the right number of microns per pixel, don't forget to cast
#  the result back to uint8
#
#            J = np.asarray(I.read_region(corner[0:2],use_level,(dx,dy)))
#            J_resize = skimage.transform.resize(J,(dx,dy),mode='constant')
#            the original code, above, yielded a dx x dy x 4 float64 array! Yikes! 
# 
#   Below I also dropped the pointless alpha layer
#
            Jraw = np.asarray(I.read_region(corner[0:2],use_level,(dx,dy)))[:,:,0:3]
            
            box_style = EMPTY_BOX
            if self.tile_has_tissue(Jraw):
                tile = dlTile(Jraw,corner,currentFile)
                txn.put(str(tileCount).encode(), pickle.dumps(tile))

                if tileCount % block == 0:
                    print(str(tileCount-tileCountZero) + '/' + str(len(ultry_corners))+" at " + self.time_of_day())
                tileCount +=1
                ul_corners.append(corner)
                box_style = TISSUE_BOX
           
            if visualize:
                plt.figure(1)
                plt.clf()
                plt.imshow(J_whole)
                
                xdraw = (corner[0] + xsq*dxscoot)/level_downs[-1]
                ydraw = (corner[1] + ysq*dyscoot)/level_downs[-1]
                
                plt.plot(xdraw,ydraw,box_style)
                plt.pause(0.05)
                plt.figure(2)
                plt.clf()
                plt.imshow(Jraw)
                plt.pause(0.01)
            
        print(env.path() + ' now has ' + str(tileCount) + ' tiles...')
        return tileCount

    def get_sourcefiles(self):
        sfiles = []
        with self.env.begin() as txn:
            sfiledict = pickle.loads(txn.get('processed files'.encode()))
        for key in sfiledict.keys():
            sfiles.append(key)
        return sfiles               
        
    def record_metadata(self,txn,currentFile,stuff):
        # print('file key is ',self.just_filename(currentFile).encode())
        txn.put(self.just_filename(currentFile).encode(), pickle.dumps(stuff))

    def get_aug_trans(self, ntiles, xysize, debug=False):
            #
            # Augment the data by doing a left-right flip, one of four 90
            # degree rotatations, and a randomly oriented stretch or 
            # compression. This last deformation is lognormal distributed, 
            # with a standard deviation of 0.2 . 
            #
            # I want to describe all these augmentations by matrix factors, 
            # so that I can build up one transformation matrix and then 
            # use it for both masks and images. 
            #
            width, height = xysize
        
            flipit = np.random.randint(0,high=2,size=ntiles)
            nrot = np.random.randint(0,high=4,size=ntiles)
            theta = np.random.uniform(low=0.0, high=2.0*np.pi,size=ntiles)
            stretch = np.random.lognormal(sigma = .2,size=ntiles)

            F = lambda f: np.asmatrix([[1-2*f,  0, width*f],\
                                       [  0,    1,    0   ],\
                                       [  0,    0,    1   ]])
            
            dx = [0, 0, width, width]
            dy = [0, height, height,0]
            
            
            C90 = np.abs(2-np.asarray([0,1,2,3])) - 1  # cos(n pi/2)
            S90 = np.roll(C90,1)
            R90n = lambda n: np.asmatrix([[ C90[n],  S90[n], dx[n]],\
                                          [-S90[n],  C90[n], dy[n]],\
                                          [  0,        0,   1]])
            
            R = lambda tht: np.matrix([[np.cos(tht), -np.sin(tht), 0],\
                                       [np.sin(tht),  np.cos(tht), 0],\
                                       [    0,             0,      1]])
            S = lambda strch : np.asarray([[strch,0,0],[0,1,0],[0,0,1]])
            T = lambda fi,nr,tht,strch: \
            np.matmul(np.matmul(np.matmul(R(-tht),np.matmul(S(strch),R(tht))),\
                                R90n(nr)),F(fi))
            Tinv = lambda fi,nr,tht,strch : np.linalg.inv(T(fi,nr,tht,strch))
            # need to loop and return ntiles x 3 x 3
    
            if debug:
                for i in range(ntiles):
                    print('flip,nrot,theta,stretch = ',flipit[i],nrot[i],\
                          theta[i],stretch[i])
            
            aminv = np.zeros((ntiles,3,3))
            for i in range(ntiles):
                aminv[i,:,:] = Tinv(flipit[i],nrot[i],theta[i],stretch[i])
                
            return aminv


    def feed_pytorch(self,N=None,maskfile=None,augment=False):

        if type(N) is list:
            tiles = self.get_tile_by_number(N)
            (nx,ny,nz) = tiles[0].data.shape
            inbatch = np.ndarray((len(tiles),nx,ny,3))
            for i,tile in enumerate(tiles):
                inbatch[i,:,:,:] = tile.data[:,:,0:3]
            ntiles = len(tiles)
        else:
            if N is None:
                N = 100
            inbatch = self.get_minibatch(N)
            ntiles = N
            #nsamp, nx, ny, nz) = inbatch.shape
        
        inbatch = (inbatch - np.mean(inbatch))/np.std(inbatch)
        
        if augment:
            augmatinv = self.get_aug_trans(ntiles,(nx,ny))
            for i in range(ntiles):
                inbatch[i,:,:,:] = warp(inbatch[i,:,:,:], augmatinv[i,:,:],\
                       mode='reflect')

        if not maskfile == None:
            maskbatch = np.zeros_like(inbatch)
            I = openslide.open_slide(self.input_directory + '/' + maskfile)
            if len(I.level_dimensions) > 1:
                print('Careful! Using level zero for masks may be incorrect!')
            for i,tile in enumerate(tiles):
                mask = np.asarray(I.read_region(tile.origin[0:2], 0, (nx,ny)))
                maskbatch[i,:,:,:] = mask[:,:,0:3]

            if augment:# need both data and masks
                for i in range(ntiles):
                    maskbatch[i,:,:,:] = warp(maskbatch[i,:,:,:], augmatinv[i,:,:],\
                           mode='reflect')
            maskbatch = np.transpose(maskbatch,axes=[0,3,1,2])

 
        inbatch = np.transpose(inbatch[:,:,:,0:3],axes=[0,3,1,2])

        if not maskfile == None:
            return torch.from_numpy(inbatch).float(), maskbatch
        else:
            return torch.from_numpy(inbatch).float()
        
        
        
    
    def feed_caffe(self, caffe_name, test_frac):     
        
        label = '0'   # YIKES!!! No classification labels! 
        
        train_txt  = caffe_name + self.get_slash() + 'train.txt'
        test_txt = caffe_name + self.get_slash() + 'test.txt'
        
        with self.env.begin() as txn:
            with txn.cursor() as curs:
                if curs.first():
                   while True:
                       tile = pickle.loads(txn.get(curs.key()))
                       (x0,y0)=tile.origin
                       last_dot = tile.sourcefile.rfind('.')
                       last_slash = tile.sourcefile.rfind(self.get_slash())
                       filename = tile.sourcefile[last_slash:last_dot] + \
                       '_' + str(x0) + '_' + str(y0) + '.jpg'
                       
                       full_path = caffe_name + filename
                       #sp.misc.imsave(full_path, tile.data[:,:,0:3])
                       imageio.imwrite(full_path, tile.data[:,:,0:3])
                       
                       text_entry = full_path + '   ' + label + '\n' # all labels zero for now
                       
                       if (np.random.uniform() > test_frac):
                          with open(train_txt,'a') as f:
                              f.write(text_entry)
                       else:
                          with open(test_txt,'a') as f:
                              f.write(text_entry)

                       
                       if not curs.next():
                           break
                else:
                    print('Something is very wrong...first key is bad...can''t even start...')
            
            
    def lmdbget(self,key):
        try:
            value = pickle.loads(self.env.begin().get(str(key).encode()))
        except:
            print(key + ' not found in database...')
            value = None
        return value

    def lmdbput(self,key,value):
        try:
            with self.env.begin(write=True) as txn:
                txn.put(key.encode(),pickle.dumps(value),overwrite=True)
        except:
            print('OOPS, can\'t write to database...')
        return
            
    def stat(self):
        return self.env.stat()
    
    def get_tile_by_number(self,tilenum):
        if not type(tilenum) == list:
            tilenum = [tilenum]
        tilenum.sort()
        
        tiles = []
        foundtiles = []
        with envr.begin(write=False) as txn:
            TI = self.lmdbget('tile index')

        for key, value in TI.items():
            nlist = [n for n in tilenum if n >= value[0] if n < value[1]]
            if len(nlist) > 0:
                with lmdb.open(key).begin() as txn:
                    for n in nlist:
                        tiles.append(pickle.loads(txn.get(str(n).encode())))
                        foundtiles.append(n)
        
        [print('tile ' + str(n) + ' not found.') for n in tilenum if n not in foundtiles];
        
        return tiles
        
    def get_random_tile(self):  
        with envr.begin(write=False) as txn:
            num_tiles_str = 'number_of_tiles'.encode()
            npick = int(np.floor(
                np.random.uniform(low=0,high=pickle.loads(txn.get(num_tiles_str)))))
            keypick = str(npick).encode()
            TI = self.lmdbget('tile index')
        
        for key, value in TI.items():
            if npick >= value[0] and npick < value[1]:
                with lmdb.open(key).begin() as txn:
                    tile = pickle.loads(txn.get(keypick))
                    break
        
        return tile

    def get_minibatch(self,N):
        mbt = self.get_random_tile()
        (nx, ny, nalpha) = mbt.data.shape
        batch = np.zeros((N,nx,ny,nalpha))
        for i in range(N):
            batch[i,:,:,:] = self.get_random_tile().data
        return batch
            
    
    def tile_has_tissue(self, J): # this works for H & E only!!!
        white = np.min(J,axis=2)> (0.92 * 255)
        return np.mean(white) < 0.9  
    
    def uichoosedir(self): # not even kidding
        from tkinter.filedialog import FileDialog
        root = tk.Tk()
        root.focus_force()
        root.withdraw() # we don't want a full GUI, so keep the root window 
                        #  from appearing
        pathname = tk.filedialog.askdirectory()
        return pathname
    
    def DLDBchoosefile(self):
        import tkinter as tk
        from tkinter.filedialog import FileDialog
        root = tk.Tk()
        root.withdraw() # we don't want a full GUI, so keep the root window from appearing
        filename = tk.filedialog.askopenfilename()
        return filename

    def get_slash(self):
        if platf.system() == 'Windows':
            slash = '\\' # So pythonic!! Duplicit is better than complicit. 
        else:
            slash = '/'
        return slash
    
    def date_for_filename(self):
        tgt = time.localtime()
        year = str(tgt.tm_year)
        mon = "{:02}".format(tgt.tm_mon)
        day = "{:02}".format(tgt.tm_mday)
        hour = "{:02}".format(tgt.tm_hour)
        minute = "{:02}".format(tgt.tm_min)
        datestr = year + mon + day + '_' + hour + minute
        return datestr

    def time_of_day(self):
        tgt = time.localtime()
        hour = "{:02}".format(tgt.tm_hour)
        minute = "{:02}".format(tgt.tm_min)
        timestr = hour + ':' + minute
        return timestr
        
    def just_filename(self, path):
        return path.split(sep=self.get_slash())[-1]
        
    
class dlTile(DLDB): 
    # A tile is a numpy image of the right shape, with the source file name
    # and upper-left-corner coordinates attached
    def __init__(self,img,corner,currentFile):
        dx, dy = np.array(img[:,:,0]).shape
        self.data = img
        self.origin = (corner[0],corner[1])
        self.segment = corner[2]
        self.sourcefile = super().just_filename(currentFile)
        
    def show(self):
#            plt.close()
#            plt.figure()
#            plt.clf()
        plt.imshow(self.data)
        plt.title(super().just_filename(self.sourcefile)+ ': ' + str(self.origin))
        plt.show()
       
    def get_metadata(self,key=None):
        with envr.begin(write=False) as txn:
            md = pickle.loads(txn.get(self.sourcefile.encode()))
        if key==None:
            for key,value in md.items():
                try:
                    if type(md[key])==str:
                        mlist = md[key].split(sep=',')
                        if len(mlist) > 1:
                            md[key] = mlist[self.segment]
                except BaseException as e:
                    pass
                
                            
            return md
        elif key=='?':
            return([key for key in md.keys()])
        else:
            return(md[key])
        

    def show_metadata(self):
        md = self.get_metadata()
        for key in md.keys():
            print(key,': ',md[key])


#------------------------------------------------------------------------
                
#    import matplotlib.pyplot as plt
#    import numpy as np
#    import openslide
#    import DLDB as dl  # argh no that is not how this works. How do I import DLDB so that 
#                 # my hand segmenter is a subclass or child class of it? Is 
                 # subclass and child class the same thing? 
    
    
    # something like class TissueSeg(DLDB) goes here. The __init__ can basically be pulled 
    # from the __main__ bit below.
    
    class TissueSeg():
        
        def GPchoosefile(self):
            import tkinter as tk
            from tkinter.filedialog import FileDialog
            root = tk.Tk()
            root.withdraw() 
            filename = tk.filedialog.askopenfilename()
            return filename
 
        def onclick(self, event):
            if event.xdata == None:
                plt.pause(0.5)
                plt.close()
            
            if event.button == 3:
                self.xpoly.append(self.xpoly[0])
                self.ypoly.append(self.ypoly[0])
                plt.plot(self.xpoly,self.ypoly)
                self.fig.canvas.draw()
                this_poly = list(zip(np.rint(self.xpoly).astype(int),np.rint(self.ypoly).astype(int)))
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
                                    
            J_whole = np.asarray(I.read_region((0,0),len(level_dims)-1,level_dims[-1]))   
            plt.imshow(J_whole)
            plt.show()
            
            self.fig = plt.gcf()
            #ax = plt.gca()
        
            self.xpoly = []
            self.ypoly = []
            
            self.polygons = []
            
            self.file = imageFile
            
            self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)


#
# http://geomalgorithms.com/a03-_inclusion.html
#
# routine for performing the "point in polygon" inclusion test

# Copyright 2001, softSurfer (www.softsurfer.com)
# This code may be freely used and modified for any purpose
# providing that this copyright notice is included with it.
# SoftSurfer makes no warranty for this code, and cannot be held
# liable for any real or imagined damage resulting from its use.
# Users of this code must verify correctness for their application.

# translated to Python by Maciej Kalisiak <mac@dgp.toronto.edu>

#   a Point is represented as a tuple: (x,y)

#===================================================================

# is_left(): tests if a point is Left|On|Right of an infinite line.

#   Input: three points P0, P1, and P2
#   Return: >0 for P2 left of the line through P0 and P1
#           =0 for P2 on the line
#           <0 for P2 right of the line
#   See: the January 2001 Algorithm "Area of 2D and 3D Triangles and Polygons"

def is_left(P0, P1, P2):
    return (P1[0] - P0[0]) * (P2[1] - P0[1]) - (P2[0] - P0[0]) * (P1[1] - P0[1])

#===================================================================

#===================================================================

# wn_PnPoly(): winding number test for a point in a polygon
#     Input:  P = a point,
#             V[] = vertex points of a polygon
#     Return: wn = the winding number (=0 only if P is outside V[])

def wn_PnPoly(P, V):
    wn = 0   # the winding number counter

    # repeat the first vertex at end
#    V = tuple(V[:]) + (V[0],) # no, I already did so in WSIseg()

    # loop through all edges of the polygon
    for i in range(len(V)-1):     # edge from V[i] to V[i+1]
        if V[i][1] <= P[1]:        # start y <= P[1]
            if V[i+1][1] > P[1]:     # an upward crossing
                if is_left(V[i], V[i+1], P) > 0: # P left of edge
                    wn += 1           # have a valid up intersect
        else:                      # start y > P[1] (no test needed)
            if V[i+1][1] <= P[1]:    # a downward crossing
                if is_left(V[i], V[i+1], P) < 0: # P right of edge
                    wn -= 1           # have a valid down intersect
    return wn

def in_poly(P,V):
    return(wn_PnPoly(P,V) != 0)



#Comments saved from above:
#
#
#------------------
# Here I need to handle the cases where there is more than one species on a slide.
# Basic plan is to look at species field and comma-separate it. If there is more than
# one entry, I need to hand segment the by specifiying polygons enclosing each piece. 
# So there's one polygon per species or tissue type, and therefore I can pair the right
# species with each tile. THis means also that "UniqueID" is no longer good enough
# to determine the full metadata dictionary of the tile. I guess maybe there just needs
# to be a get_species and get_tissue methods, and these access the UniqueID'd 
# dictionary and pick the right species. The tile will need a segment number or 
# polygon number generated by using the tile origin and the file dict's polygon list. 
#
# Ok, it's different from that a bit: I need to re-write the metadata so that by 
# the time one gets to this point in DLDB, the polygon list is already available. 
# So I should probably write another function that fixes the metadata. The following 
# writes a dataframe to Excel. 
#
#  writer = pd.ExcelWriter('output.xlsx')
#  df1.to_excel(writer,'Sheet1')
#  df2.to_excel(writer,'Sheet2')
#  writer.save()
# Then I need to learn to append a column to a dataframe. 
#------------------
