import torch
import os
import pandas as pd
import wfdb

leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

class PretrainDataset(torch.utils.data.Dataset):
    def __init__(self, config, leads_to_use=leads, split='train', augmentations=None):
        self.leads = leads if leads_to_use == ['*'] else leads_to_use
        self.patch_size = config.patch_size
        self.split = split
        self.augmentations = augmentations
        self.need_second_transform = config.use_teacher_student and config.strategy == 'masked_token_prediction'

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = str(self.records[idx])

        s, _ = wfdb.rdsamp(os.path.join(self.data_folder, record, record))
        s = torch.tensor(s, dtype=torch.float32)

        if self.leads != leads:
            # keep the selected leads
            s = s[:, [leads.index(lead) for lead in self.leads]].squeeze()

        if self.augmentations is not None:
            signal = self.augmentations(s)
        else:
            signal = s

        if self.need_second_transform and self.augmentations is not None: 
            # apply the second transform
            signal_2 = self.augmentations(s)
        
            return {
                'signal': signal,
                'signal_2': signal_2,
            }
        else:
            return {
                'signal': signal,
            }
            