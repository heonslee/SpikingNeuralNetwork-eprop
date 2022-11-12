#import matplotlib.pyplot as plt
import numpy as np
#from time_hs import *
#%%
def templates2clusters(tmpl_waveforms, spk_tmpl, spk_clst, use_clst, spk_times):
    # Note that the output "spk_clst1" is not ID of phy-gui.
    # spk_times MUST be in "ms" unit
    import sys
    n_tmpl, n_time, n_ch = tmpl_waveforms.shape
    # if n_tmpl != max(spk_tmpl)+1:
        # print("number of templates is not consistent!")
        # sys.exit(1)
    n_clst = len(use_clst)
    len_times = len(spk_times)
    
    clst_waveforms = np.zeros((n_clst, n_time, n_ch))
    clst2tmpl_mat = np.zeros((n_clst, n_tmpl))
    n_spikes = np.zeros((n_clst,))
    i_time1 = np.zeros((len_times, n_clst), dtype=bool) # ****
    # (1) Find corresponding templates
    for i, clst in enumerate(use_clst):
        i_time1[:,i] = spk_clst == clst # *** the index where spikes of cluster occur
        n_spikes[i] = sum(1*i_time1[:,i])
        TMPLs = np.unique(spk_tmpl[i_time1[:,i]])
        print("cluster  ", clst, " has templates,", TMPLs)
        if len(TMPLs)==0:
            continue
        for tmpl in TMPLs:
            i_time2 = spk_tmpl[i_time1[:,i]] == tmpl
            clst2tmpl_mat[i, tmpl] = sum(1*i_time2)/sum(1*i_time1[:,i])
        
        W = clst2tmpl_mat[i, :]
        tmpl_waveforms_tmp = np.moveaxis(tmpl_waveforms, 0, 1)
        clst_waveforms[i,:,:] = np.dot(W, tmpl_waveforms_tmp)

    # (2) Construct new spike times (1-d, only used clusters)
    i_time_use = np.any(i_time1, axis=1)
    spk_times1 = spk_times[i_time_use]
    spk_clst1 = spk_clst[i_time_use]
    for i, clst in enumerate(use_clst):
        ind = spk_clst1 == clst
        spk_clst1[ind] = i
    
    # (3) Construct new spike times (2d-array form)
    n_spikes = n_spikes.astype(int)
    spk_times2 = np.zeros((n_spikes.max(), n_clst))*np.nan
    for i, clst in enumerate(use_clst):
        spk_times2[0:n_spikes[i],i] = spk_times[i_time1[:,i]]
        
    # (4) Firing rate for all periods
    tmax_in_sec = spk_times[-1]/1000 # ms => sec
    frate_allperiods = n_spikes/tmax_in_sec
    return clst2tmpl_mat, clst_waveforms, spk_times1, spk_clst1, spk_times2, frate_allperiods
#%% Construct new spike times (2d-array form)
def GetSpkTimes2(spktr, fs=1.0):
    # Unit is "ms"
    # !!!fs: data / ms < == if 1ms resolution, then fs=1
    # spktr shape = [tsize, n_nrn]
    n_spikes = np.nansum(spktr,0).astype(int)
    n_spikes_total = np.sum(n_spikes)
    n_nrn = np.shape(spktr)[1]
    
    spk_times2 = np.zeros((n_spikes.max(), n_nrn))*np.nan
    # spk_times1 = np.empty((1,n_nrn),dtype = object)
    # spk_times1 = [None]*n_nrn
    spk_times0 = np.zeros(n_spikes_total,)
    for nrn in range(n_nrn):
        i_spikes = np.where(spktr[:,nrn] >=1)[0] # ***
        if len(i_spikes)!=0:
            spk_times2[0:n_spikes[nrn],nrn] = i_spikes
            # spk_times1[nrn] = i_spikes
    clst_vec = np.tile(np.arange(0, n_nrn).reshape((1,n_nrn)),  (np.shape(spk_times2)[0],1))
    spk_times2_tmp = spk_times2.reshape(-1,)
    clst_vec = clst_vec.reshape(-1,)
    
    ind = ~np.isnan(spk_times2_tmp)
    spk_times0 = spk_times2_tmp[ind]
    spk_clst0 = clst_vec[ind]
    
    ind = np.argsort(spk_times0)
    spk_times0 = spk_times0[ind]
    spk_clst0 = spk_clst0[ind]
    
    spk_times2 /= fs
    # spk_times1 /= fs
    spk_times0 /= fs
    spk_times1 = np.nan
    
    return spk_times1, spk_times2, spk_times0, spk_clst0
#%% Construct new spike times (2d-array form)
def GetSpkTimes2fast(spktr, fs=1.0):
    # Unit is "ms"
    # !!!fs: data / ms
    # spktr shape = [tsize, n_nrn]
    n_spikes = np.nansum(spktr,0).astype(int)
    n_nrn = np.shape(spktr)[1]
    
    spk_times2 = np.zeros((n_spikes.max(), n_nrn))*np.nan
    for nrn in range(n_nrn):
        i_spikes = np.where(spktr[:,nrn])[0] # ***
        if len(i_spikes)!=0:
            spk_times2[0:n_spikes[nrn],nrn] = i_spikes
    spk_times2 /= fs
    return spk_times2
#%% Construct new spike times (2d-array form)
def GetSpkTimes2_tr(spk_train_tr):
    n_spikes = np.sum(spk_train_tr,0).astype(int)
    n_nrn, n_trials = np.shape(n_spikes)
    max_spikes = n_spikes.max()
    
    spk_times2_tr = np.zeros((max_spikes, n_nrn, n_trials))*np.nan # ***
    n_spikes_tr = np.zeros((n_trials, n_nrn), dtype = int)
    for j in range(n_trials):
        for nrn in range(n_nrn):
            i_spikes = np.where(spk_train_tr[:,nrn, j] >=1)[0] # ***
            spk_times2_tr[0:n_spikes[nrn,j],nrn,j] = i_spikes
            n_spikes_tr[j,nrn] = len(i_spikes)
    return spk_times2_tr, n_spikes_tr
#%%
def FindChannel(clst_waveforms):
    n_clst, n_time, n_ch = clst_waveforms.shape
    ch_clst = np.zeros((n_clst,))*np.nan
    for nrn in range(n_clst):
        if np.count_nonzero(clst_waveforms[nrn,:,:])!=0:
            waveforms = np.squeeze(clst_waveforms[nrn,:,:])
            ch_clst[nrn] = np.argmin(np.min(waveforms, axis=0))
    return ch_clst
#%%
def WaveformFeatures(clst_waveforms, *t):
    n_clst, n_time, n_ch = clst_waveforms.shape
    if t not in locals() and t not in globals():
        t = np.linspace(0, 3, n_time)
        
    wfeatures = np.zeros((n_clst, 8))*np.nan
    for nrn in range(n_clst): # nrn = 0
        waveform_tmp = clst_waveforms[nrn,::]  
        if np.count_nonzero(waveform_tmp)==0:
            continue
        
        waveform_min = np.amin(waveform_tmp, axis=0)
        ch_template = np.argmin(waveform_min)
        waveform_tmp = waveform_tmp[:,ch_template]
        
        i_trough = waveform_tmp.argmin()
        v_trough = waveform_tmp[i_trough]
        i_peak1 = waveform_tmp[1:i_trough].argmax()
        v_peak1 = waveform_tmp[i_peak1]
        i_peak2 = waveform_tmp[i_trough:].argmax()
        i_peak2 = i_peak2+i_trough-1
        v_peak2 = waveform_tmp[i_peak2]
        # 1) trough
        # 2) height
        v_height = max(v_peak1, v_peak2) - v_trough    
        # 3) asymmetric index
        v_asym = (v_peak2-v_peak1)/(v_peak2+v_peak1)    
        # 4) peak to peak time
        t_p2p = t[i_peak2] - t[i_peak1]    
        # 5) trough to right peak
        t_t2p = t[i_peak2] - t[i_trough]    
        # 6) left peak to trough
        t_p2t = t[i_trough] - t[i_peak1]    
        # 7) asymmetry in area
        i_pos1 = [j for j,x in enumerate(waveform_tmp) if x>=0 and j<i_trough]
        i_pos2 = [j for j,x in enumerate(waveform_tmp) if x>=0 and j>i_trough]
        a_pos1 = sum(waveform_tmp[i_pos1])
        a_pos2 = sum(waveform_tmp[i_pos2])
        a_asym =  abs((a_pos1-a_pos2)/(a_pos1+a_pos2))
        a_asym =  (a_pos1-a_pos2)/(a_pos1+a_pos2)
    #    a_asym =  abs(a_pos1-a_pos2)
    #    a_asym =  abs((a_pos1-a_pos2)/v_trough)
        a_asym = abs((a_pos1+a_pos2)/v_trough)
        #8) half-width
        i_hwidth1 = np.argmin(np.abs(waveform_tmp[:i_trough] - v_trough/2))
        i_hwidth2 = np.argmin(np.abs(waveform_tmp[i_trough:] - v_trough/2))
        t_halfwidth = t[i_trough+i_hwidth2] - t[i_hwidth1]
        
        wfeatures[nrn,:] = [v_trough, v_height, v_asym, t_p2p, t_t2p, t_p2t, a_asym, t_halfwidth]
    return wfeatures
#%%
def getSpikeTrain(spk_times2, tmax_ms):
    _, n_nrn = spk_times2.shape
    tmax_ms = np.ceil(tmax_ms).astype(int)
    spk_train = np.zeros((tmax_ms,n_nrn), dtype = int)
    tt = np.arange(0, tmax_ms, 1)
    
    for nrn in range(n_nrn): # nrn = 0
        spk_times2_tmp = spk_times2[:,nrn]
        bool_nan = np.isnan(spk_times2_tmp)
        if sum(bool_nan*1)!=0:
            i_nan1 = np.where(bool_nan)[0][0]
            spk_times2_tmp = spk_times2_tmp[:i_nan1]
        spk_train_tmp, _ = np.histogram(spk_times2_tmp, bins = tt) # 1ms time bin ***
        spk_train[:-1,nrn] = spk_train_tmp # value can be larger than 1  !!!! ***
    return spk_train
#%%
def times2train_1(tSpk, tmax_ms):
    tmax_ms = np.ceil(tmax_ms).astype(int)
    tt = np.arange(-.5, tmax_ms+.5, 1)
    spktr, _ = np.histogram(tSpk, bins = tt) # 1ms time bin ***
    return spktr




