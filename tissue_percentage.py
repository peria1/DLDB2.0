#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 06:31:42 2018

@author: bill
"""
import openslide
import numpy as np
import glob
import billUtils as bu
import pickle

"""
    Given a color micrograph of some Hematoxylin and Eosin stained tissue in 
    H_and_E, this function computes the fraction of the micrograph that contains tissue. 
    
    The criterion "white" is just an empirical test for the absence of tissue, i.e. 
    the background is white. I don't test for proper H & E colors. So far it has
    seemed pretty safe. 
    
"""

def get_tissue_percentage(H_and_E):
    nx, ny, nz = H_and_E.shape
    
    minc = np.min(H_and_E[:,:,0:3],axis=2)
    maxc = np.max(H_and_E[:,:,0:3],axis=2)
    
    white = minc > (.92*255)
    white_frac = np.nansum(white)/np.prod(H_and_E.shape[0:2])
    
    sat_img = (maxc - minc)/maxc
    sat = np.nanmean(sat_img[white == 0])
    
    return int(100-np.round(white_frac*100)), int(np.round(sat*100))
        

if __name__ == "__main__":

    openslide.Image.MAX_IMAGE_PIXELS = None # prevents DecompressionBomb Error

    dirs_to_process = bu.uichoosedirs()
    print(dirs_to_process)
    file_dict = {}
    for dir_to_process in dirs_to_process:
        files_to_process = sorted(glob.glob(dir_to_process+'/*.tif'))
        files_to_process = [f for f in files_to_process if '_cpd' not in f]
        print(files_to_process)
        for file_to_process in files_to_process:
            Tissue = openslide.open_slide(file_to_process)
            allT = np.asarray(Tissue.read_region((0,0),0,Tissue.dimensions))[:,:,0:3]    
            pct, sat = get_tissue_percentage(allT)
            
            this_file = bu.just_filename(bu, file_to_process)
            print('{:s}: {:3d} tissue , {:3d} saturated'.format(this_file, pct, sat))
            file_dict.update({this_file : (pct , sat)})
    
    output_name = 'tissue_pct_' + bu.date_for_filename() + '.dat'
    with open(output_name,'wb') as f:
        f.write(pickle.dumps(file_dict))
            
            
        
