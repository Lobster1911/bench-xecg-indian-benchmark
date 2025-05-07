import torch
import os
import pandas as pd
import wfdb

leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

class PretrainDataset(torch.utils.data.Dataset):
    def __init__(self, config, leads_to_use=leads, split='train', global_augmentations=None, local_augmentations=None):
        self.leads = leads if leads_to_use == ['*'] else leads_to_use
        self.patch_size = config.patch_size
        self.split = split
        self.global_augmentations = global_augmentations
        self.local_augmentations = local_augmentations
        self.n_global_view = config.n_global_view
        self.n_local_view = config.n_local_view

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = str(self.records[idx])

        s, _ = wfdb.rdsamp(os.path.join(self.data_folder, record, record))
        s = torch.tensor(s, dtype=torch.float32)

        if self.leads != leads:
            # keep the selected leads
            s = s[:, [leads.index(lead) for lead in self.leads]].squeeze()

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