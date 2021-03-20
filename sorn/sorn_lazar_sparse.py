import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
sys.path.append(r'C:\Users\yowch\Dropbox\mypy\functions')
# sys.path.append(r'E:\Dropbox (Personal)\mypy\functions')
from time_hs import *
import visual_hs as vh
import numba as nb
import bct
vh.defaultFigOptions2()
#%% function
@nb.jit(nopython=True)
# @nb.jit(nopython=True, parallel=True) # This is slower for small network, but faster for large network
def sorn_lazar(u_ext,ksi,thresh,nunmu,wee_v,wee_tf,wei_v,wei_tf,wie_v,wie_tf):
    tsize,nE = u_ext.shape
    n_nrn = ksi.shape[1]
    nI = n_nrn - nE
    n_ee,n_ei,n_ie = wee_tf.shape[0],wei_tf.shape[0],wie_tf.shape[0]
    nu_stdp,nu_istdp,nu_ip,mu_ip = nunmu[0],nunmu[1],nunmu[2],nunmu[3]
    wee_sn,wei_sn = np.ones((nE),dtype=nb.float64),np.ones((nE),dtype=nb.float64)
    wee_sn_update,wei_sn_update = wee_sn.copy(),wei_sn.copy()
    xe,xi = np.zeros((tsize,nE), dtype=nb.int32),np.zeros((tsize,nI), dtype=nb.int32)
    xe_tmp,xi_tmp = np.zeros((nE),dtype=nb.float64),np.zeros((nI),dtype=nb.float64)    
    thresh_e,thresh_i = np.zeros((tsize,nE),dtype=nb.float64), thresh[nE:].copy() # !!! You made mistake here
    thresh_e[0] = thresh[:nE].copy()
    ksi_e,ksi_i = ksi[:,:nE].copy(), ksi[:,nE:].copy()
    for i in np.arange(1, tsize-1): # i=1    i+=2, tt+=.5
        # (1) E=>E
        xe_tmp[:] , wee_sn_update[:] = 0., 0.
        for j in range(n_ee):
            wee_v[j] = wee_v[j]/wee_sn[wee_tf[j,0]] # 1) synaptic normalization (using 1step past data)
            xe_tmp[wee_tf[j,0]] += wee_v[j]*xe[i,wee_tf[j,1]] # 2) synaptic input
            wee_v[j] += nu_stdp*(xe[i,wee_tf[j,0]]*xe[i-1,wee_tf[j,1]] - xe[i-1,wee_tf[j,0]]*xe[i,wee_tf[j,1]]) # 3,4) STDP
            wee_v[j] = wee_v[j]*(wee_v[j]>0)
            wee_sn_update[wee_tf[j,0]] += wee_v[j] # 5) update normalization term
        wee_sn = wee_sn_update.copy()       
        
        # (2) I=>E
        for j in range(n_ei):
            xe_tmp[wei_tf[j,0]] += wei_v[j]*xi[i,wei_tf[j,1]] # 2) synaptic input
        
        # (3) IP update
        thresh_e[i] = thresh_e[i-1] + nu_ip*(xe[i] - mu_ip)
        # thresh_e[i][thresh_e[i]<0] = 0. # !!! this is worse??? it seems like, yes...no?
        
        # (4) E neuron update
        xe_tmp += u_ext[i] + ksi_e[i] - thresh_e[i]
        xe[i+1] = xe_tmp>0
        
        # (5) I neuron update (E=>I)
        xi_tmp[:]=0.
        for j in range(n_ie):
            xi_tmp[wie_tf[j,0]] += wie_v[j]*xe[i,wie_tf[j,1]]
        xi_tmp += ksi_i[i] - thresh_i
        xi[i+1] = xi_tmp>0
        
    x = np.column_stack((xe,xi))
    return x, wee_v, wei_v, thresh_e, wee_sn, wei_sn
#%% 1-1) define time
time,dt = 500000,10
t = np.arange(0, time, dt)
tsize = t.shape[0]
#%% 1-2) input data
import torch
# from torchvision import transforms
# from bindsnet.network import Network, nodes, topology, monitors
ipt_strength = 1. # !!!
words = np.array([[1,2,2,2,2,3],[4,5,5,5,5,6]],dtype=int) # !!!
words -= 1
letts = np.unique(words) # = [0,1,2,3,4,5]
lett_num = len(letts) # = 6
word_num = words.shape[0] # =2
n_repeat = 4
# n_patshown = int(tsize/lett_num)

pat1,pat2 = np.zeros((n_repeat+2,lett_num)),np.zeros((n_repeat+2,lett_num)) # pat1 shape=[6,6], pat2 shape=[6,6]
for l in range(lett_num):
    pat1[l,words[0][l]] = ipt_strength
    pat2[l,words[1][l]] = ipt_strength
n_patshown = int(tsize/pat1.shape[0])
data_12 = np.random.choice(2,n_patshown, replace=True) # 0 or 1
labels, labels_pat1, labels_pat2 = np.zeros((0),dtype=int), np.where(pat1!=0)[1], np.where(pat2!=0)[1] # labels_pat1=[0,1,1,1,1,2], labels_pat2=[3,4,4,4,4,5]
labels2_name = ['a','b1','b2','b3','b4','c','d','e1','e2','e3','e4','f']
labels2, labels2_pat1,labels2_pat2 = np.zeros((0),dtype=int), np.arange(lett_num),lett_num+np.arange(lett_num) # labels1_pat1=[0,1,2,3,4,5], labels2_pat=[6,7,8,9,10,11]
data_tmp = np.zeros((0, lett_num))
for i in range(n_patshown):
    if data_12[i]==0: # if 0 (i.e., [a,b1,b2,b3,b4,c])
        data_tmp = np.vstack((data_tmp, pat1))
        labels = np.concatenate((labels, labels_pat1))
        labels2 = np.concatenate((labels2, labels2_pat1))
    else: # if 1 (i.e., [d,e1,e2,e3,e4,f])
        data_tmp = np.vstack((data_tmp, pat2))
        labels = np.concatenate((labels, labels_pat2))
        labels2 = np.concatenate((labels2, labels2_pat2))
labels = labels[1:] # !!!
    

if tsize > data_tmp.shape[0]:
    time = data_tmp.shape[0]*dt
    t = np.arange(0, time, dt)
    tsize = t.shape[0]
#%% 1-3) network parameters
nE=800 #=> ~40 seconds for 50,000 steps
# nE = 200
nI, n_nrn = int(nE*.2), nE + int(nE*.2)
lambda_w = 10
w_ee = bct.makerandCIJ_dir(nE, nE*lambda_w) # !!!
# w_ee = bct.makerandCIJ_dir(nE, int(nE*(nE-1)*.1)) # Random network .05 !!!
w_ee[w_ee!=0] = np.random.rand(np.sum(w_ee!=0))
w_ie = np.random.rand(nI, nE) # e=>i
w_ei = -np.random.rand(nE, nI) # i=>e
w_ii = np.zeros((nI,nI))
w_ee /= np.sum(w_ee,1).reshape(-1,1) + 1e-300 # synaptic normalization
w_ei /= np.sum(np.abs(w_ei),1).reshape(-1,1) + 1e-300 # synaptic normalization
w_ie /= np.sum(w_ie,1).reshape(-1,1) + 1e-300 # synaptic normalization
w = np.row_stack((np.column_stack((w_ee, w_ei)), np.column_stack((w_ie, w_ii)) ))
w0 = w.copy()
# plt.imshow(w)

n_ext = int(nE*.1/lett_num) # !!! input recieving neurons per letter (Del Papa) 
# n_ext = int(nE*.05/lett_num) # !!! input recieving neurons per letter (Lazar)
# n_ext = lett_num*n_ext # !!!
data = np.zeros((data_tmp.shape[0],0))
for c in range(lett_num):
    data = np.hstack((data, np.tile(data_tmp[:,c].reshape(-1,1),(1,n_ext))   ))
u_ext = np.zeros((tsize, nE))
u_ext[:,:n_ext*lett_num] = data

# ksi = np.random.normal(0, np.sqrt(0.05), (tsize, n_nrn)) # !!! Del Papa
ksi = np.zeros((tsize, n_nrn)) # !!! Lazar
# ksi[:,:nE] = .01 # !!! DC current

# thresh = np.concatenate((np.random.rand(nE), 0.5*np.random.rand(nI))) # !!! Del Papa
thresh = np.concatenate((0.5*np.random.rand(nE), np.random.rand(nI))) # !!! Lazar

# nunmu = np.array([4*1e-3,1e-3,1e-2,1e-1]) # !!! Del Papa (learning rates for stdp, istdp, ip, mu_ip)
# nunmu = np.array([1e-3,0,1e-3,2*(n_ext*lett_num)/nE]) # !!! Lazar, better...?
nunmu = np.array([1e-3,0,1e-3,2*(n_ext)/nE]) # !!! Lazar(2)
#%% 1-4) Make network sparse
from scipy.sparse import csr_matrix
wee = csr_matrix(w_ee)
wee_v = wee.data
wee_to, wee_from = wee.nonzero()
wee_tf = np.column_stack((wee_to, wee_from))
wee_mat = w_ee.copy()

wei = csr_matrix(w_ei)
wei_v = wei.data
wei_to, wei_from = wei.nonzero()
wei_tf = np.column_stack((wei_to, wei_from))
wei_mat = w_ei.copy()

wie = csr_matrix(w_ie)
wie_v = wie.data
wie_to, wie_from = wie.nonzero()
wie_tf = np.column_stack((wie_to, wie_from))
wie_mat = w_ie.copy()
#%% 1-5) Run simulation
ttt = tic()
_,_,_,_,_,_ = sorn_lazar(u_ext[:3],ksi,thresh,nunmu,wee_v,wee_tf,wei_v,wei_tf,wie_v,wie_tf)
toc(ttt)
ttt = tic()
x,wee_v,wei_v,thresh_e,wee_sn,wei_sn = sorn_lazar(u_ext,ksi,thresh,nunmu,wee_v,wee_tf,wei_v,wei_tf,wie_v,wie_tf)
x = x[:-1] # because the last can't be predicted !!!
toc(ttt)

wee.data,wei.data = wee_v.copy(),wei_v.copy()
w_ee_new,w_ei_new = csr_matrix.todense(wee),csr_matrix.todense(wei)
w[:nE,:] = np.hstack((w_ee_new, w_ei_new))
#%% plot
import spikes4 as spk4
import datamanipulation as dm
nrn=np.array([lett_num*n_ext+1,lett_num*n_ext+2])
fig,axes = plt.subplots(4,3, figsize=(15,8),constrained_layout=True)
spk4.rasterplot(x[:1000,:300], 'train', fs=100/1000, ax=axes[0,0])
axes[1,0].plot(t[:1000], dm.smooth(np.nanmean(x[:1000,2*n_ext:],1),15))
axes[2,0].plot(t[:1000], thresh_e[:1000,nrn])

spk4.rasterplot(x[-1000:,:300], 'train', fs=100/1000, ax=axes[0,1], t0=t[-1000])
axes[1,1].plot(t[-1000:], dm.smooth(np.nanmean(x[-1000:,lett_num*n_ext:nE],1),15))
axes[2,1].plot(t[-1000:], thresh_e[-1000:,nrn])

axes[3,0].imshow(w0[:nE,:],vmin=-0.01,vmax=0.01,cmap='bwr')
axes[3,1].imshow(w[:nE,:], vmin=-0.01,vmax=0.01,cmap='bwr')
#%%****************************************************************************
#%% 2,3) ANN for readout
import torch.nn as nn
class NN(nn.Module):
    def __init__(self, input_size, n_classes):
        super(NN, self).__init__()
        self.linear_1 = nn.Linear(input_size, n_classes)
    def forward(self, x):
        # out = torch.sigmoid(self.linear_1(x.float().view(-1))) # this can recieve 1 data only
        out = torch.sigmoid(self.linear_1(x.float())) # to receive batch data
        return out
gpu = False
n_classes = lett_num - 2
model = NN(n_nrn, n_classes) # !!! this is better?

if gpu:
    device="cuda: 0"
    model.to(device)
# criterion = torch.nn.MSELoss(reduction="sum") # MSE
# criterion = torch.nn.MSELoss(size_average=True) # MSE
# criterion = torch.nn.BCELoss(size_average=True) # BCE (bad...)
criterion = torch.nn.BCELoss(reduction="sum") # !!! BCE (best)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#%% 2-1) Traning readout (freezing weight )
print('running with freezed network')
ttt=tic()
nunmu[:3] = 0
i_rt = np.random.randint(tsize-5101)
x_tr,_,_,_,_,_ = sorn_lazar(u_ext[i_rt:i_rt+5100],ksi[i_rt:i_rt+5100],thresh,nunmu,wee_v,wee_tf,wei_v,wei_tf,wie_v,wie_tf)
toc(ttt)
labels_tr = labels[i_rt:i_rt+5100]

x_tr = x_tr[100:]
labels_tr = labels_tr[100:]
#%% 2-2) Organize input data (5000 steps)
import pytorchhelp
import torch.utils.data as data_utils
labels_tmp = labels_tr.copy().astype(float)# exclude transition point & change label
replace = {0:np.nan, 1:0, 2:1, 3:np.nan, 4:2, 5:3}# exclude transition point & change label
for k,v in replace.items(): 
    labels_tmp[labels_tr==k] = v
ind = np.where(~np.isnan(labels_tmp))[0]
labels_tr = labels_tmp[ind].astype(int)
x_tr = torch.tensor(x_tr[ind])
print('label nums: %d' %(len(np.unique(labels_tr))))

x_tr, labels_tr = pytorchhelp.balance_classes(x_tr, labels_tr, batch_size=None, randomize=True) # !!!

labels_1hot = np.squeeze(np.eye(n_classes)[labels_tr.reshape(-1)])
labels_1hot = torch.from_numpy(labels_1hot).type(torch.FloatTensor)
if gpu:
    labels_1hot = labels_1hot.to(device)
batch_size = 50 # using batch !!!
dataset = pytorchhelp.gencustomdatasets(x_tr, labels_1hot)
data_loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
#%% 2-3) Training ANN
n_epochs = 40
n_crr_tr, loss_avg = np.zeros(n_epochs), np.zeros(n_epochs)
y = np.zeros((len(labels_tr)))
for epoch in range(n_epochs): # epoch=0
    ttt = tic()
    for step,batch in enumerate(data_loader): # spktr_tmp,j= spktrE[0],0
        optimizer.zero_grad() # initialize optimizer
        outputs = model(batch[0]) # (1) Forward
        loss = criterion(outputs, batch[1]) # (2) Loss function
        loss_avg[epoch] += loss.data
        loss.backward() # (3) Backword
        optimizer.step() # (4) Optimize
        n_crr_tr[epoch] += sum(outputs.argmax(dim=1) == batch[1].argmax(dim=1)).float()/batch[1].shape[0]
    toc(ttt)
acc_tr = n_crr_tr/(step+1)
#%% 2-4) plot
axes[0,2].plot(acc_tr)
axes[1,2].plot(loss_avg)
#%% 3-1) Testing readout (freezing weight)
print('running with freezed network')
ttt=tic()
nunmu[:3] = 0
i_rt = np.random.randint(tsize-5101)
x_ts,_,_,_,_,_ = sorn_lazar(u_ext[i_rt:i_rt+5100],ksi[i_rt:i_rt+5100],thresh,nunmu,wee_v,wee_tf,wei_v,wei_tf,wie_v,wie_tf)
toc(ttt)
labels_ts = labels[i_rt:i_rt+5100]
labels2_ts = labels2[i_rt:i_rt+5100]

x_ts = x_ts[100:]
labels_ts = labels_ts[100:]
labels2_ts = labels2_ts[100:]
#%% 3-2) Organize input data (5000 steps)
labels_tmp = labels_ts.copy().astype(float)# exclude transition point & change label
replace = {0:np.nan, 1:0, 2:1, 3:np.nan, 4:2, 5:3}# exclude transition point & change label
for k,v in replace.items(): 
    labels_tmp[labels_ts==k] = v
ind = np.where(~np.isnan(labels_tmp))[0]
labels_ts = labels_tmp[ind].astype(int)
x_ts = torch.tensor(x_ts[ind])
labels2_ts = labels2_ts[ind]
# labels2_ts = [labels2_ts[i] for i in ind]
print('label nums: %d' %(len(np.unique(labels_ts))))
# x_ts, labels_ts = pytorchhelp.balance_classes(x_ts, labels_ts, batch_size=None, randomize=True) # !!!

labels_1hot = np.squeeze(np.eye(n_classes)[labels_ts.reshape(-1)])
labels_1hot = torch.from_numpy(labels_1hot).type(torch.FloatTensor)
if gpu:
    labels_1hot = labels_1hot.to(device)
batch_size = 50 # using batch !!!
dataset = pytorchhelp.gencustomdatasets(x_ts, labels_1hot)
data_loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
#%% 3-3) Testing readout
n_crr_ts = 0
for step,batch in enumerate(data_loader): # spktr_tmp,j= spktrE[0],0
    optimizer.zero_grad() # initialize optimizer
    outputs = model(batch[0]) # (1) Forward
    loss = criterion(outputs, batch[1]) # (2) Loss function
    loss_avg[epoch] += loss.data
    loss.backward() # (3) Backword
    optimizer.step() # (4) Optimize
    n_crr_ts += sum(outputs.argmax(dim=1) == batch[1].argmax(dim=1)).float()/batch[1].shape[0]
acc_ts = n_crr_ts/(step+1)
axes[0,2].plot(epoch, acc_ts,'r*')
#%% 3-4) plot
confusion_matrix = torch.zeros(n_classes, n_classes)
with torch.no_grad():
    for j,x_ts_tmp in enumerate(torch.tensor(x_ts)):
        outputs = model(x_ts_tmp)
        preds = torch.argmax(outputs)
        confusion_matrix[labels_ts[j], preds]+=1
print(confusion_matrix)
axes[2,2].imshow(confusion_matrix,cmap='hot')
#%% PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 4)
pca.fit(x_ts)
x_pca = pca.fit_transform(x_ts)
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
print(np.unique(labels2_ts))
print(np.unique(labels2_pat1))
labels2_pat12 = np.concatenate((labels2_pat1, labels2_pat2))
print(np.unique(labels2_pat12))
print(labels2_name)

labels2_pat12 = np.array([0,1,2,3,4,6,7,8,9,10],dtype=int)
# colors = sns.color_palette('husl', n_colors=lett_num)  # 1 word
colors = sns.color_palette('hls', n_colors=len(labels2_pat12))  # 2 words

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
for j,l in enumerate(labels2_pat1): # plot only pat1 (a,b,b,b,c)
    ind = labels2_ts==l
    ax.plot(x_pca[ind,0],x_pca[ind,1],x_pca[ind,2], '.', color=colors[j], label=labels2_name[l])
ax.legend()
#%%
# import pandas as pd
# df_pca = pd.DataFrame(data=x_pca, columns=['1','2','3','4'])
# df_pca['l2'] = labels2_ts
# # df_pca = pd.melt(df_pca,  value_vars=['1','2','3'])
# # plt.figure()
# # sns.pairplot(df_pca, hue='variable')

# colors = sns.color_palette('hls', n_colors=12)  # a list of RGB tuples

# sctmplot = pd.plotting.scatter_matrix(df_pca[['1','2','3','4']],   c = [colors[j] for j in labels2_ts], 
#                                       marker = '.', hist_kwds={'bins':40},  figsize=(8,8), alpha=.2)

