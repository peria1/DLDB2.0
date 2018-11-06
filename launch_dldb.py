#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 13:44:01 2018

@author: bill
"""

import dldb
import billUtils as bu

in_dir = bu.uichoosedir(title='Choose input directory...')
out_dir = bu.uichoosedir(title='Choose output directory...')

db = dldb.DLDB(input_directory=in_dir,build=True,\
               tileSize=(256,256),use_level=0, \
               output_directory =out_dir)

