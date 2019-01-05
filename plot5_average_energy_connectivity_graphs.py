#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 13:03:32 2018

@author: oliver
"""

import scipy as sp
import numpy as np
#import pylab as pl
import networkx as nx
#from math import factorial
#from multiprocessing import Pool
#import dask.array as	da
import multiprocessing
#from networkx.algorithms.connectivity import local_node_connectivity# checar si la necesito
#from networkx.algorithms.flow import shortest_augmenting_path
#import itertools
#from networkx.algorithms.connectivity import (build_auxiliary_node_connectivity)
#from networkx.algorithms.flow import build_residual_network
####Random graphs, Small world 10 nodes#####

#def comb2(n, k):
#    return factorial(n) / factorial(k) / factorial(n - k)
##########################################################################

def AvconducBA(N):
    #kneigh=6
    #graph=nx.barabasi_albert_graph(N,kneigh)
    prob=0.5
    graph=nx.erdos_renyi_graph(N,prob)
    #avconn=nx.average_node_connectivity(graph)
    lapmat=nx.laplacian_matrix(graph).todense()
    AvDegree=lapmat.diagonal().mean()
    eigvecs=sp.linalg.eigh(lapmat,turbo=True)[1]
    NMfrac=int(len(lapmat)*1.0)
    count=0
    lista=[]
    for iresidue in range(N):
        for jresidue in range(iresidue+1,N):
            if np.logical_and(lapmat[iresidue,iresidue]>AvDegree,lapmat[jresidue,jresidue]>AvDegree)==True:
                QQ=0.
                for k in range(NMfrac):
                    xi_i = np.abs(eigvecs[iresidue,k])
                    xi_j = np.abs(eigvecs[jresidue,k])
                    if xi_i != 0 and xi_j != 0:
                        QQ +=(xi_i*xi_j/(xi_i + xi_j))
                lista.append(QQ)
                count +=1
    lista=np.array(lista)
    Average=lista.mean()
    
    return N,Average,count,AvDegree


############################################################################
if __name__ == '__main__':
    n_times = 10
    pool = multiprocessing.Pool(processes=16)
    results = pool.map(AvconducBA,[2000]*n_times )

results=np.asarray(results)
fname_template="NodesandEnergy_selected_pairs_connectivity_BA2000.dat"
np.savetxt(fname_template, results,delimiter=' ', newline='\n',fmt='%1.8e')