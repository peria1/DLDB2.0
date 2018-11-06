#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 11:05:41 2018

Things I can't believe are not part of a package. Maybe they are! 

@author: bill
"""
import tkinter as tk
from tkinter.filedialog import FileDialog
import time
import platform as platf

def uichoosefile(title = None):
    root = tk.Tk()
    root.withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = tk.filedialog.askopenfilename(title=title)
    return filename

def uichoosefiles(title = None):
    root = tk.Tk()
    root.withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = tk.filedialog.askopenfilenames(title=title)
    return filename


def uichoosedir(title = None):
    root = tk.Tk()
    root.focus_force()
    root.withdraw() # we don't want a full GUI, so keep the root window 
                    #  from appearing
    pathname = tk.filedialog.askdirectory(title=title)
    return pathname

def uichoosedirs(title = None):
    root = tk.Tk()
    root.focus_force()
    root.withdraw() # we don't want a full GUI, so keep the root window 
                    #  from appearing
    pathnames = []
    dirselect = tk.filedialog.Directory(title=title)
    while True:
        d = dirselect.show()
        if not d: break
        pathnames.append(d)
        
    return pathnames


def date_for_filename():
    tgt = time.localtime()
    year = str(tgt.tm_year)
    mon = "{:02}".format(tgt.tm_mon)
    day = "{:02}".format(tgt.tm_mday)
    hour = "{:02}".format(tgt.tm_hour)
    minute = "{:02}".format(tgt.tm_min)
    datestr = year + mon + day + '_' + hour + minute
    return datestr

def time_of_day():
    tgt = time.localtime()
    hour = "{:02}".format(tgt.tm_hour)
    minute = "{:02}".format(tgt.tm_min)
    timestr = hour + ':' + minute
    return timestr
    
def just_filename(self, path):
    return path.split(sep=self.get_slash())[-1]


def get_slash():
    if platf.system() == 'Windows':
        slash = '\\' 
    else:
        slash = '/'
    return slash
 
