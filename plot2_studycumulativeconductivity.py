#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 10:28:45 2018

@author: Oliver Becerra Gonzalez
"""

import scipy as sp
import numpy as np
import pylab as pl
#from math import factorial
from prody import *
import glob
from multiprocessing import Pool
from numba import jit
#import dask.array as	da
############################Defining the function that calculates energy transfer Q_ij between i,j residues
@jit(parallel=True,fastmath=True)
def Energy_current(item):
    prot = parsePDB(item)
    calphas=prot.select('protein and name CA')
    anm=ANM('ANM analysis')
    anm.buildHessian(calphas,cutoff=10.0,gamma=1)
    Hessian=anm.getHessian().round(3)
    anm.calcModes(n_modes='all',zeros=False)#
    eigvecs=anm.getEigvecs()
    aa=(len(anm)+6)/3
    Qcumcond=np.zeros(aa)
    #Q=np.zeros((aa,aa))
    for i in range((len(anm)+6)/3):
        for j in range(i+1,(len(anm)+6)/3):
            numeritoQ=0.
	    Qvals=[]
            for mode in range(len(anm)):
                inorm2=eigvecs[i,mode]**2 +eigvecs[i+1,mode]**2 + eigvecs[i+2,mode]**2 
                jnorm2=eigvecs[3*j-3,mode]**2 +eigvecs[3*j-2,mode]**2 + eigvecs[3*j-1,mode]**2 
                if inorm2 !=0 and jnorm2 !=0 :
                    numeritoQ += (inorm2*jnorm2)/(inorm2 + jnorm2)
                #Q[i,j]=numeritoQ
                #Q[j,i]=numeritoQ
		Qvals.append(numeritoQ)
	    Qvals=np.cumsum(np.asarray(Qvals))
            Qcumcond += Qvals
                #Kijcum +=Qcumul
                #lists_of_lists = [[1, 2, 3], [4, 5, 6]]
                #[sum(x) for x in zip(*lists_of_lists)]
                #Kijcum=Kijcum/comb2(N,2)
    Av_cumconduc=Qcumcond/comb2(aa,2)
    #m = Q.shape[0]
    #idx = (np.arange(1,m+1) + (m+1)*np.arange(m-1)[:,None]).reshape(m,-1)
    #out = Q.ravel()[idx]
    #out=out.reshape(out.shape[0]*out.shape[1])
    #avQIJallmodes=np.mean(out)/2
    #return aa, avQIJallmodes
    return Av_cumconduc
###############################################

########Getting the PDB's from the specific folder########3   
folder = "/home/oliver//Documents/Documents/Irregular structures/*.pdb"
file_list=glob.glob(folder)
#######################################
####Multiprocessing the script#########

if __name__ == "__main__":
    pool = Pool(16)
    results = pool.map(Energy_current,file_list)
    results=np.asarray(results)
    #N=len(file_list)
    #results=results.reshape((N,2))
#####Saving the results to a File#######################################
fname_template="AminoacidnumberandEnergy_irregularproteins.dat"
np.savetxt(fname_template, results,delimiter=' ', newline='\n',fmt='%1.8e')
