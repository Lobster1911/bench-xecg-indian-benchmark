import torch
import os
import pandas as pd
import wfdb
import neurokit2 as nk
import numpy as np

leads = ['i', 'ii', 'iii', 'avR', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']

class PretrainDataset(torch.utils.data.Dataset):
    def __init__(self, config, split='train', global_augmentations=None, local_augmentations=None):
        self.leads = leads
        self.patch_size = config.patch_size
        self.split = split
        self.global_augmentations = global_augmentations
        self.local_augmentations = local_augmentations
        self.n_global_view = config.n_global_view
        self.n_local_view = config.n_local_view
        self.sampling_freq = config.sampling_freq
        self.nk_clean = config.nk_clean

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = str(self.records[idx])

        s, info = wfdb.rdsamp(os.path.join(self.data_folder, record))

        # if nan fill
        if np.isnan(s).any():
            print("WARNING: Nan detected")
       
        # mapping leads in the correct position
        s = self.map_leads_and_clean(s, info)
        s = self.resample_if_needed(s, info)

        if self.global_augmentations is not None:
            global_signals = [ self.global_augmentations(s) for _ in range(self.n_global_view)]
        else:
            global_signals = s
        
        if self.local_augmentations is not None and self.n_local_view > 0:
            local_signals = [ self.local_augmentations(s) for _ in range(self.n_local_view)]
        else:
            local_signals = None

        return  {
            'global_signals': global_signals,
            'local_signals': local_signals,
        }
    
    def map_leads_and_clean(self, signal, info):
        s = np.zeros((len(signal), len(self.leads)))
        for lead in info['sig_name']:
            l = lead.lower()
            if l in self.leads:
                if self.nk_clean:
                    s[:, self.leads.index(l)] = nk.ecg_clean(signal[:, info['sig_name'].index(lead)], sampling_rate=info['fs']).copy()
                else:
                    s[:, self.leads.index(l)] = signal[:, info['sig_name'].index(lead)]
        return s
    
    def resample_if_needed(self, signal, info):
        if self.sampling_freq != info['fs']:
            signal = nk.signal_resample(signal, sampling_rate=info['fs'], desired_sampling_rate=self.sampling_freq, method='FFT')
        
        signal = torch.tensor(signal, dtype=torch.float32)
        return signal
                

