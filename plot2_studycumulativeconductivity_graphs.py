#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 13:18:59 2018

@author: oliver
"""

import scipy as sp
import numpy as np
import pylab as pl
import networkx as nx
from math import factorial
from multiprocessing import Pool
#from numba import jit
import dask.array as	da
import multiprocessing
####Random graphs, Small world 10 nodes#####
#iterationsnum=5
#Qvals=[]
#Kijcum=np.zeros(10)
#@jit(parallel=True,fastmath=True)
def comb2(n, k):
    return factorial(n) / factorial(k) / factorial(n - k)

#############################################################3
def ErdGraph(N):
    lista=[]
    #N=20
    #prob=0.5
    kneigh=6
    #graph=nx.erdos_renyi_graph(N,prob)
    graph=nx.barabasi_albert_graph(N,kneigh)
    lapmat=nx.laplacian_matrix(graph).todense()
    lapmat=da.from_array(lapmat,chunks=10)
    eigvecs=sp.linalg.eigh(lapmat,turbo=True)[1]
    NMfrac=int(len(lapmat)*1.0)
    Qcumcond=np.zeros(N)
    for iresidue in range(N):
        for jresidue in range(iresidue+1,N):
            #QQ=0.
            Qvals=[]
            for k in range(NMfrac):
                xi_i = np.abs(eigvecs[iresidue,k])
                xi_j = np.abs(eigvecs[jresidue,k])
                if xi_i != 0 and xi_j != 0:
                    #QQ +=(xi_i*xi_j/(xi_i + xi_j))
                    Qvals.append((xi_i*xi_j/(xi_i + xi_j)))
        Qvals=np.cumsum(np.asarray(Qvals))
        Qcumcond += Qvals
	Av_cumconduc=Qcumcond/comb2(N,2)
    lista.append(Av_cumconduc)
    return lista

if __name__ == '__main__':
    n_times = 10
    pool = multiprocessing.Pool(processes=16)
    results = pool.map(ErdGraph,[20]*n_times )

results=np.asarray(results)
fname_template="Cumulative_conductivity_BA200.dat"
np.savetxt(fname_template, results,delimiter=' ', newline='\n',fmt='%1.8e')
####################################Barabasi-Albert##################33
### esta es la buena###
def BAGraph(N):
    lista=[]
    Kneigh=6
    #prob=0.5
    #graph=nx.erdos_renyi_graph(N,prob)

    graph=nx.barabasi_albert_graph(N,Kneigh)
    lapmat=nx.laplacian_matrix(graph).todense()
    lapmat=da.from_array(lapmat,chunks=10)
    eigvecs=sp.linalg.eigh(lapmat,turbo=True)[1]
    NMfrac=int(len(lapmat)*1.0)
    Qcumcond=np.zeros(N)
    for iresidue in range(N):
        for jresidue in range(iresidue+1,N):
            Qvals=[]
            for k in range(NMfrac):
                xi_i = np.abs(eigvecs[iresidue,k])
                xi_j = np.abs(eigvecs[jresidue,k])
                if xi_i != 0 and xi_j != 0:
                    Qvals.append((xi_i*xi_j/(xi_i + xi_j)))
            Qvals=np.cumsum(np.asarray(Qvals))
            Qcumcond += Qvals
	Av_cumconduc=Qcumcond/comb2(N,2)
    lista.append(Av_cumconduc)
    return Av_cumconduc#lista#KijfracAver#(np.mean(out)/2)

if __name__ == '__main__':
    n_times = 10
    pool = multiprocessing.Pool(processes=16)
    results = pool.map(BAGraph,[2000]*n_times )

results=np.asarray(results)
fname_template="Cumulative_conductivity_BA2000.dat"
np.savetxt(fname_template, results,delimiter=' ', newline='\n',fmt='%1.8e')
#######################################################3

#pool=multiprocessing.Pool()
#result2=pool.map(Energy_current,file_list)
#if __name__ == "__main__":
#    pool = Pool(16)
#    results = pool.map(WSGraph,range(iterationsnum))
#    results=np.asarray(results)
#    N=iterationsnum
#    results=results.reshape((N,2))

results=ERGraph(10)
#results=np.asarray(KijfracAver)
#results=results.reshape((10,2))
fname_template="Cumulative_conductivity_ER20.dat"
np.savetxt(fname_template, results,delimiter=' ', newline='\n',fmt='%1.8e')
##############################
results=BAGraph(100)
#results=np.asarray(KijfracAver)
#results=results.reshape((100,2))
fname_template="Cumulative_conductivity_BA20.dat"
np.savetxt(fname_template, results,delimiter=' ', newline='\n',fmt='%1.8e')
