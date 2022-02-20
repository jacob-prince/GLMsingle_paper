import numpy as np
import pandas as pd
import nibabel as nib
import os
import sys
from os import listdir
from os.path import isfile, join, exists
import scipy.io as sio
import pymatreader
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import cortex
import seaborn as sns
from IPython.core.debugger import set_trace

####################

def get_B5K_condition_indices(subj):
    
    bidsdir = join('/lab_data','tarrlab','common','datasets','BOLD5000','BIDS');

    eventdir = join(bidsdir,f'sub-{subj}')

    if 'CSI4' in subj:
        nses = 9
    else:
        nses = 15

    runtrs = 194
    tr = 2

    stim_names = []
    
    session_nruns = []

    for ses in range(1,nses+1):
        if ses < 10:
            sesstr = f'0{ses}'
        else:
            sesstr = str(ses)

        subeventdir = join(eventdir,f'ses-{sesstr}','func')

        eventfiles = []
        for ef in np.sort(os.listdir(subeventdir)):
            if np.logical_and('events.tsv' in ef, 'localizer' not in ef):
                eventfiles.append(ef)

        session_nruns.append(len(eventfiles))
        
        for run in range(len(eventfiles)):
            temp = pd.read_csv(join(subeventdir,eventfiles[run]),sep='\t')
            stim_names.append(list(temp['ImgName']))

    stim_names = np.concatenate(stim_names)
    
    print(f'subj {subj}, n stimuli = {len(stim_names)}')

    # unique_names are the unique alphabetical image filenames, len = 4916 for CSI1-3
    # cond_labels are the chronological 0-indexed indices associating each stimulus with a member of "unique_names"  
    unique_names, _, cond_labels = np.unique(stim_names,return_index=True,return_inverse=True)
    
    return unique_names, cond_labels, session_nruns, stim_names


#####################

def get_NSD_condition_indices():
    
    nsesimgs = 750
    nses = 10
    nsesruns = 12
    
    a1 = sio.loadmat('/user_data/jacobpri/Project/BOLD5000-GLMs/manuscript/nsd_expdesign.mat')
    
    labels = np.squeeze(a1['masterordering'])[:nsesimgs*nses]

    sesidx = np.concatenate([np.ones((nsesimgs,)) * r for r in range(nses)])
    
    session_nruns = np.array([nsesruns for r in range(nses)])

    assert(len(sesidx) == len(labels))
    
    return labels, session_nruns

   

######################

def load_reliability_dict(info, tag):
    
    labdatadir = '/lab_data/tarrlab/jacobpri/BOLD5000-GLMs/betas'
    userdatadir = '/user_data/jacobpri/Project/BOLD5000-GLMs/betas'
    
    reliability = dict()
    
    # load nsdgeneral masks
    masks = load_nsdgeneral_masks()

    # check to be sure all the datafiles exist
    for ds in list(info.keys()):

        reliability[ds] = dict()

        for subj in info[ds]['subjs']:

            subj_version_list = []

            reliability[ds][subj] = dict()

            methods = [key for key in list(info[ds].keys()) if 'subj' not in key]

            for method in methods:

                basedir,methodstr,version = info[ds][method][0], info[ds][method][1], info[ds][method][2] 
                
                if tag == 'reliability':
                    betadir = f'{basedir}/{methodstr}/{subj}'
                else:
                    betadir = f'{userdatadir}/{methodstr}/{subj}'

                metric_savefn = f'{betadir}/quality_metrics/{subj}_{version}_{tag}.npy'

                rel = np.load(metric_savefn)

                mask = masks[ds][subj]==1

                if np.ndim(rel) == 3:
                    rel = rel[mask==1]

                #print(ds,method,subj,rel.shape)

                reliability[ds][subj][method] = rel

                subj_version_list.append(rel)

            reliability[ds][subj]['mean'] = np.mean(np.stack(subj_version_list,axis=1),axis=1)
            
    return reliability

#####################

def load_nsdgeneral_masks():
    
    #print('loading nsdgeneral masks...')
    
    maskdir = '/lab_data/tarrlab/jacobpri/BOLD5000-GLMs/masks'
    masks = dict()
    
    # iterate through datasets
    for ds in ['NSD','B5K']:
    
        masks[ds] = dict()
        
        if ds == 'NSD':
            subjs = [f'subj0{n}' for n in range(1,5)]
        else:
            subjs = [f'CSI{n}' for n in range(1,5)]

        # iterate through subjects
        for subj in subjs:

            masks[ds][subj] = nib.load(f'{maskdir}/{subj}_nsdgeneral.nii.gz').get_data()
            
    #print('\tdone.')
    
    return masks 

#######################

def plot_flat_map(dataset, subj, vol, title = '', cmap = 'spectral_r', vmin = 0, vmax = 0.6, colorbar = True):
    
    vol_reshape = np.transpose(vol,(2,1,0))
    
    if dataset == 'B5K':
        xfm = 'full'
        subj = f'sub-{subj}'
    else:
        xfm = 'func1pt8_to_anat0pt8_autoFSbbr'
        
    vol_data = cortex.Volume(vol_reshape, subj, xfm, cmap=cmap, vmin=vmin, vmax=vmax)
    
    #plt.figure(figsize=(12,8));
    cortex.quickflat.make_figure(vol_data, with_rois=False, with_labels=False, with_colorbar=colorbar);
    plt.title(title, fontsize = 24)
    
    return vol_data


#########################

def reshape_nsdgeneral_to_volume(data_vector, mask_3d):
    
    # get 3d indices for each item in masked vector
    idx_flat = np.ravel_multi_index(np.where(mask_3d == 1), mask_3d.shape) #FLAT 1D indexes using original mask
    idx_2d = np.unravel_index(idx_flat, mask_3d.shape)   #2D INDEXES using the FLAT 1D indexes

    out = np.full(mask_3d.shape,np.nan)

    for i in range(idx_2d[0].shape[0]):
        out[idx_2d[0][i], idx_2d[1][i], idx_2d[2][i]] = data_vector[i]
        
    return out

########################

def translatevmetric(x):
    
    f = 1 - x ** 2
    
    f[f<0] = 0
    
    f = np.sqrt(f) / x
    
    f[np.isinf(f)] = np.nan
    
    return f