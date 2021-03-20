import numpy as np
import scipy.spatial.distance
from scipy.stats import norm
import matplotlib.pyplot as plt

def distance_bin(A):
    # think of the meaning of A @ A
    #
    # A: adjacency matrix
    # d: distance matrix (the output)
    # n_path: for undetermined elements, this represents the existence of connectivity (connected? or disconnected?)
    #   thus, whether the elements of n_path is 0 or not is important (we don't care exact values...)
    # L: L=n_path, but counts only the elements where d[elements]=0
    #   that is, L is the number of "undetermined" paths, at each iteration
    
    n = 1 # n represents the path length (it starts with 1)
    d = np.eye(A.shape[0]) # distance matrix, starting with an identity matrix
    n_path = A.copy() # starting with a copy of A
    L = n_path!=0 # L is the number of "undetermined" paths
    while len(np.flatnonzero(L))>=1:
        d += n*L
        n += 1 # n represents the path length (it starts with 1)
        n_path = n_path@A # for undetermined elements, this represents the "existence of connectivity"
        L = (n_path!=0) * (d==0) # update L, but only the undetermined elements!!!
    
    d[d==0] = np.inf # disconnected edges => Length=infinity
    # np.fill_diagonal(d, 0)
    d -= np.eye(A.shape[0])
    return d
    
def gen_randA_und(N,K,connectedness=True):
    ind = np.triu(1-np.eye(N,dtype=int),1)
    ind_r, ind_c = np.where(ind)
    
    A = np.zeros((N,N),dtype=int)
    i_random = np.random.permutation(len(ind_r))
    ind_r2,ind_c2 = ind_r[i_random],ind_c[i_random]
    ind_r2,ind_c2 = ind_r2[:K], ind_c2[:K]
    A[ind_r2,ind_c2] = 1
    A = A + A.T
    
    # ensure connectedness
    if connectedness:
        while any(np.sum(A,0)==0):
            A = np.zeros((N,N),dtype=int)
            i_random = np.random.permutation(len(ind_r))
            ind_r2,ind_c2 = ind_r[i_random],ind_c[i_random]
            ind_r2,ind_c2 = ind_r2[:K], ind_c2[:K]
            A[ind_r2,ind_c2] = 1
            A = A + A.T
    return A
    
def gen_randA_dir(N,K,connectedness_in=True, connectedness_out=False):
    ind = 1-np.eye(N,dtype=int)
    ind_r,ind_c = np.where(ind)
    
    A = np.zeros((N,N),dtype=int)
    i_random = np.random.permutation(len(ind_r))
    ind_r2,ind_c2 = ind_r[i_random], ind_c[i_random]
    ind_r2,ind_c2 = ind_r2[:K], ind_c2[:K]
    A[ind_r2, ind_c2] = 1

    # ensure connectedness
    if connectedness_in and connectedness_out:
        while any(np.sum(A,1)==0) or any(np.sum(A,0)==0):
            A = np.zeros((N,N),dtype=int)
            i_random = np.random.permutation(len(ind_r))
            ind_r2,ind_c2 = ind_r[i_random], ind_c[i_random]
            ind_r2,ind_c2 = ind_r2[:K], ind_c2[:K]
            A[ind_r2, ind_c2] = 1
    elif connectedness_in:
        while any(np.sum(A,1)==0):
            A = np.zeros((N,N),dtype=int)
            i_random = np.random.permutation(len(ind_r))
            ind_r2,ind_c2 = ind_r[i_random], ind_c[i_random]
            ind_r2,ind_c2 = ind_r2[:K], ind_c2[:K]
            A[ind_r2, ind_c2] = 1
    elif connectedness_out:
        while any(np.sum(A,0)==0):
            A = np.zeros((N,N),dtype=int)
            i_random = np.random.permutation(len(ind_r))
            ind_r2,ind_c2 = ind_r[i_random], ind_c[i_random]
            ind_r2,ind_c2 = ind_r2[:K], ind_c2[:K]
            A[ind_r2, ind_c2] = 1
    return A    
#################################################
def aij_distance2d(xysize, sig, nE, nI, weight, cself=False, plot=False, randomseed=None):
    ''' directionality: column to row index '''
    # xysize,nE,nI,sig =  [2000, 1000], 80, 20, [250,150]
    # xysize,nE,nI,sig =  [20000, 5000], 8000, 2000, [250,150]
    np.random.seed(randomseed)
    
    
    n_nrn = nE + nI
    x = xysize[0]*np.random.rand(n_nrn)
    y = xysize[1]*np.random.rand(n_nrn)
    iE = np.argsort(x[:nE])
    iI = np.argsort(x[nE:])+nE
    x = x[np.concatenate((iE,iI))]
    y = y[np.concatenate((iE,iI))]
    xy = np.column_stack((x,y))
#    en_sig3 = np.round(nE*sig[0]*5/xysize[0])
    
    gE0 = norm.pdf(0,0,sig[0])
    gI0 = norm.pdf(0,0,sig[1])    
    
#    plt.scatter(x,y)
    # E <-> E
    ind = np.triu_indices(nE, 1)
    d = scipy.spatial.distance.pdist(xy[:nE]) # 1d array
    g = norm.pdf(d,0,sig[0])
    aij_1 = (np.random.rand(g.shape[0])< g/gE0).astype(float)
    aij_2 = (np.random.rand(g.shape[0])< g/gE0).astype(float)
    aijEE_tmp1,aijEE_tmp2 = np.zeros((nE,nE),dtype=float),np.zeros((nE,nE),dtype=float)
    aijEE_tmp1[ind] = weight[0,0]*aij_1
    aijEE_tmp2[ind] = weight[0,0]*aij_2
    aijEE = aijEE_tmp1 + aijEE_tmp2.T
    
    # I <-> I
    ind = np.triu_indices(nI, 1)
    d = scipy.spatial.distance.pdist(xy[nE:]) # 1d array
    g = norm.pdf(d,0,sig[1])
    aij_1 = (np.random.rand(g.shape[0])< g/gI0).astype(float)
    aij_2 = (np.random.rand(g.shape[0])< g.T/gI0).astype(float)
    aijII_tmp1,aijII_tmp2 = np.zeros((nI,nI),dtype=float),np.zeros((nI,nI),dtype=float)
    aijII_tmp1[ind] = weight[1,1]*aij_1
    aijII_tmp2[ind] = weight[1,1]*aij_2
    aijII = aijII_tmp1 + aijII_tmp2.T
       
    # E <-> I
    # the cij matrix is by definition, j=>i (different from Pangyu's & different from Bindsnet)
    
    d = scipy.spatial.distance.cdist(xy[:nE],xy[nE:]) # 2d array (e.g., 80-by-20)
    g = norm.pdf(d,0,sig[1]) # 2d array (e.g., 80-by-20)
    aijEI = (np.random.rand(g.shape[0],g.shape[1])< g/gI0).astype(float) # I -> E (e.g., 80-by-20)
    g = norm.pdf(d,0,sig[0])
    aijIE = (np.random.rand(g.shape[1],g.shape[0])< g.T/gE0).astype(float) # E -> I (e.g., 20-by-80)
    aijEI = weight[0,1]*aijEI # I=>E (e.g., 80-by-20)
    aijIE = weight[1,0]*aijIE # E=>I (e.g., 20-by-80)
    
    aij_upper = np.column_stack((aijEE, aijEI))
    aij_lower = np.column_stack((aijIE, aijII))
    aij = np.row_stack((aij_upper, aij_lower))
    
    if cself is True:
        np.fill_diagonal(aij, np.concatenate((weight[0,0]*np.ones(nE),weight[1,1]*np.ones(nI))) ) # self connection
    cdens = np.sum(aij)/(n_nrn*(n_nrn-1))
    if plot is True:
        plt.figure()
        plt.imshow(aij,cmap='jet')
    return aij, xy, cdens





