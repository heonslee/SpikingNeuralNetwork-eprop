import numpy as np
import matplotlib.pyplot as plt
import spikes1 as spk1



def rasterplot(x, tort='train', fs=0, markersize=3, ax=None, t0=0, color='k', nrn0=0):
    ''' 
    fs: 
        e.g., time resolution=10ms => fs=0.1 => t*10
        e.g., time resolution=2ms => fs=0.5 => t*2
        e.g., time resolution=1ms => fs=1 => t*1
    tort: is input type train or time?
    nrn0: start neuron index
    '''
    if tort == 'train': # if x is spike train
        # time by n_nrn
        _, _, spkt, spkc = spk1.GetSpkTimes2(x) # unit is "ms"
        tmax, n_nrn = x.shape
    elif tort == 'time': # if x is spike time & cluster 
        spkt = x[0]
        spkc = x[1]
        n_nrn = spkc.max()
        tmax = spkt[-1]
        
    if fs != 0:
        spkt = spkt/fs
        tmax = tmax/fs
    
    spkt += t0
    tmax += t0
    spkc += nrn0
    
    if ax==None:
        plt.scatter(spkt, spkc, c=color, marker = 's', s = markersize)
        ax = plt.gca()
    else:
        ax.scatter(spkt, spkc, c=color, marker = 's', s = markersize)
    ax.set_xlim(t0,tmax)
    ax.set_ylim(nrn0-.5, nrn0+n_nrn-.5)
    
    return ax


def rasterplotLines(x, tort = 'train', fs = 0, linewidth=.5, color='k'):
    if tort == 'train': # if x is spike train
        _, _, spkt, spkc = spk1.GetSpkTimes2(x)
    elif tort == 'time': # if x is spike time & cluster 
        spkt = x[0]
        spkc = x[1]
    if fs !=0:
        spkt = spkt/fs
    for nrn in range(np.nanmax(spkc)):
        plt.vlines(spkt[spkc ==nrn], spkc[spkc == nrn], spkc[spkc == nrn]+1, color=color, linewidth=linewidth)
    return plt.gca()

def firingrateplot(spktr, fs=0, winsize=10, ax=0, vmin=0, vmax=5):
    import datamanipulation as dm
    # time by n_nrn, # time-unit is ms
    datsize,n_nrn = spktr.shape
    spktr = dm.windowing2(spktr.T, winsize) # win, time, neuron
    spktr = np.nanmean(spktr,1)
    if fs==0:
        fs=1
    tlen = spktr.shape[0]*winsize/fs
    t = np.linspace(0, tlen, spktr.shape[0])
    if ax==0:
        plt.pcolormesh(t,np.arange(n_nrn),spktr.T,vmin=vmin,vmax=vmax,cmap='gray')
    else:
        ax.pcolormesh(t,np.arange(n_nrn),spktr.T,vmin=vmin,vmax=vmax,cmap='gray')
    return plt.gca()

    
    