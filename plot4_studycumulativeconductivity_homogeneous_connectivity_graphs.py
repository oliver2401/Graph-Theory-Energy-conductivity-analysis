#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 12:35:10 2018

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
#from networkx.algorithms.connectivity import local_node_connectivity
#from networkx.algorithms.flow import shortest_augmenting_path
#import itertools
#from networkx.algorithms.connectivity import (build_auxiliary_node_connectivity)
#from networkx.algorithms.flow import build_residual_network
####Random graphs, Small world 10 nodes#####
#iterationsnum=5
#Qvals=[]
#Kijcum=np.zeros(10)
#@jit(parallel=True,fastmath=True)
#def comb2(n, k):
#    return factorial(n) / factorial(k) / factorial(n - k)
#############################################
#connectivity= degree, asi q solo hay q utilizar la degree matrix,y ver q 
#residuos tienennuna conectividad mayor q el promedio, algo parecido como con el plot 3
#acuerdate de borrar todos los archivos del plot 4 para ER y BA
###############################################################
def BAGraph(N):
    Kneigh=6
    graph=nx.barabasi_albert_graph(N,Kneigh)
    #prob=0.5
    #graph=nx.erdos_renyi_graph(N,prob)
    lapmat=nx.laplacian_matrix(graph).todense()
    AvDegree=lapmat.diagonal().mean()
    #lapmat=da.from_array(lapmat,chunks=10)
    eigvecs=sp.linalg.eigh(lapmat,turbo=True)[1]
    NMfrac=int(len(lapmat)*1.0)
    Qcumcond=np.zeros(N)
    count=0.
    for iresidue in range(N):
        for jresidue in range(iresidue+1,N):
            if np.logical_and(lapmat[iresidue,iresidue]>AvDegree,lapmat[jresidue,jresidue]>AvDegree)==True:
                Qvals=[]
                for k in range(NMfrac):#
                    xi_i = np.abs(eigvecs[iresidue,k])#
                    xi_j = np.abs(eigvecs[jresidue,k])#
                    if xi_i != 0 and xi_j != 0:#
                        Qvals.append((xi_i*xi_j/(xi_i + xi_j)))#
                count +=1.
                Qvals=np.cumsum(np.asarray(Qvals))
                Qcumcond += Qvals
    Av_cumconduc=Qcumcond/count#comb2(N,2)
    #lista.append(Av_cumconduc)
    return Av_cumconduc,count#lista#KijfracAver#(np.mean(out)/2)

if __name__ == '__main__':
    n_times = 10
    pool = multiprocessing.Pool(processes=16)
    results = pool.map(BAGraph,[2000]*n_times )


cumcond=[]
pairs=[]
for i in range(n_times):
    cumcond.append(results[i][0])
    pairs.append(results[i][1])

results_arrays=np.asarray(cumcond)#cada row es el vector cumulativo
results_pairs=np.asarray(pairs)
fname_template="Cumulative_conductivity_plot4_BA2000.dat"
np.savetxt(fname_template, results_arrays,delimiter=' ', newline='\n',fmt='%1.8e')

fname_template2="number_pairs_plot4_BA2000.dat"
np.savetxt(fname_template2, results_pairs,delimiter=' ', newline='\n',fmt='%1.8e')