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
import numpy as np

def uichoosefile(title = None, initialdir = None):
    root = tk.Tk()
    root.withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = tk.filedialog.askopenfilename(title=title, initialdir = initialdir)
    return filename

def uichoosefiles(title = None, initialdir = None):
    root = tk.Tk()
    root.withdraw() # we don't want a full GUI, so keep the root window from appearing
    filename = tk.filedialog.askopenfilenames(title=title, initialdir = initialdir)
    return filename


def uichoosedir(title = None, initialdir = None):
    root = tk.Tk()
    root.focus_force()
    root.withdraw() # we don't want a full GUI, so keep the root window 
                    #  from appearing
    pathname = tk.filedialog.askdirectory(title=title, initialdir = initialdir)
    return pathname

def uichoosedirs(title = None, initialdir = None):
    root = tk.Tk()
    root.focus_force()
    root.withdraw() # we don't want a full GUI, so keep the root window 
                    #  from appearing
    pathnames = []
    dirselect = tk.filedialog.Directory(title=title, initialdir = initialdir)
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
 
#-----------------------

def residualSymmTest(dy, x, zero=0.0, showPlot=True):
    if showPlot:
        import matplotlib.pyplot as plt
#
# Tests if residuals dy have systematic variation, by testing if the
# x-distribution of non-positive residuals is the same as the x-distribution of
# non-negative residuals, using the Kuiper two-sample test. 
#
#
# If a residual is zero, it is included in both the non-negative and
# non-positive populations. But this should be very rare, so a warning is
# issued if it occurs at all, along with a report of how many times it occurred.
# 
# Returns the hypothesis test result h, the p value, and the actual Kuiper
# statistic kp.
# 
# If the model fits, and there is no apparent additional systematic
# variation in the residuals, then h will be zero, p will be of order 1,
# and you are on your own if you wish to interpret kp. 
#
# If there is still systematic variation in the residuals, indicating that
# a more complex model is needed, then h will be 1, p will be miniscule,
# and kp will be...well, bigger than it otherwise would have been!
#
# In my experience thus far, when I fit data to polynomials of
# progressively higher order, there was nearly always an abrupt upward jump
# in p at some order, so I just stopped there. 
#
    if not x:
        x = np.linspace(0, dy.size, num=dy.size, endpoint=False)
        dy = np.reshape(dy, (dy.size))
    
    
    if x.shape != dy.shape:
        print(\
      'residual and independent variable must have same dimensions...')
        return None
    
    if not((dy > zero).any() and  (dy < zero).any()):
        print('residuals do not span ', str(zero), '...')
    
    xplus  = x(dy >= zero)
    xminus = x(dy <= zero)
    
    if (dy == 0).any():
        pass
#        print('residual was actually zero at ', int(np.sum(dy == 0)),' out of ', dy.size, ' points...\n'])
    
    
    if np.exp(np.abs(np.log(xplus.size/xminus.size))) > 2:
        print('Large skew...xplus has ', xplus.size,'elements and xminus has', xminus.size,  'elements...\n')

#
#%Finally...the actual test!
#
    h, p, kp = kscirc(xplus, xminus)
    if showPlot:
        plt.plot(x,dy,'o')
        plt.title('p = ' + str(p))
        xl = plt.xlim()
        plt.hlines(zero, xl[0], xl[1])
    
    return h, p, kp



#
# This does the Kuiper version of the KS test, which pretends that the
# range of the samples is actually a circle, i.e. that the points at plus
# and minus infinity are joined. The advantage is that the test is now
# equally sensitive throughout the sample range, whereas kstest2, for
# example, is most sensitive to midrange differences in PDF. 
#
#  

def kscirc(x1in,x2in,alpha=0.05):

    def Qkp(L):
    
        if L < 0.4:
            return 1
         
        qkpl = 0
        a2 = -2*np.power(L,2)
        for j in range(10):
            a2j2 = a2*np.power((j+1),2)
            term = 2 * (-2*a2j2-1)*np.exp(a2j2)
            qkpl = qkpl + term
        
        return qkpl
    
    x1 = x1in[np.isfinite(x1in)]
    x2 = x2in[np.isfinite(x2in)]
    x1 = np.ndarray.flatten(x1, order='K')
    x2 = np.ndarray.flatten(x2, order='K')
    
    n1 = x1.size
    n2 = x2.size
    
    all_x = np.concatenate((x1,x2))
    ii = np.argsort(all_x)
    s = all_x[ii]
    
    from1 = ii <=n1
    dF = [1/n1 if f1 else -1/n2 for f1 in from1]
    #
    # shift dFs to the right succesively, for ties, so that the net jump for
    # all occurences of each tied group occurs all at once, on the last
    # occurrence 
    #
    ties = s[2:] == s[1:-1]
    nties = np.sum(ties)
    if nties > 0:
        ities = [i for (i,t) in enumerate(ties) if t]
        for i in range(nties):
            dF[ities[i]+1] = dF[ities[i]+1] + dF[ities[i]]
            dF[ities[i]] = 0
            
    
    F = np.cumsum(dF)
    
    ks = np.max(F) - np.min(F)
    
    Ne = n1*n2/(n1 + n2)
    sqNe = np.sqrt(Ne)
    p = Qkp((sqNe + 0.155 + 0.24/sqNe)*ks)
    
    h = p < alpha
    
    return h, p, ks


    
