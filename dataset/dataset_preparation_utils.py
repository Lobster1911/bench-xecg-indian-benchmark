import wfdb
import neurokit2 as nk
import numpy as np
import os
import shutil
import h5py
import json
import simple_icd_10
import dataset.mit_bih as mit_bih
import dataset.code_dataset as code
import dataset.mimic_iv as mimic
import dataset.ptb_xl as ptb_xl
import dataset.chapman as chapman
import dataset.incart as incart
from torch.utils.data import Subset, ConcatDataset
from dataset.generic_utils import get_transforms
import torch

def load_datasets(config):

    datasets_pretrain = []
    val_datasets = []

    for dataset in config.pretrain_datasets:
        if dataset == 'mimic':
            datasets_pretrain.append(mimic.ECGMIMICDataset(
                config, 
                split='train', 
                global_augmentations=get_transforms(config, split='train', type='global'), 
                local_augmentations=get_transforms(config, split='train', type='local')
            ))
            val_datasets.append(mimic.ECGMIMICDataset(
                config, 
                split='val', 
                global_augmentations=get_transforms(config, split='train', type='global'), # I want the training augmentation in this case
                local_augmentations=get_transforms(config, split='train', type='local')
            ))
        elif dataset == 'incart':
            # only training because small sample size
            incart_dataset = incart.ECGIncartDataset(
                config, 
                split='train', 
                global_augmentations=get_transforms(config, split='train', type='global'), 
                local_augmentations=get_transforms(config, split='train', type='local')
            )
            datasets_pretrain.append(incart_dataset)
            
        elif dataset == 'code15':
            code15 = code.ECGCODE15Dataset(
                config, 
                global_augmentations=get_transforms(config, split='train', type='global'), 
                local_augmentations=get_transforms(config, split='train', type='local')
            )
            # split the dataset into train and val
            train_size = int(0.9 * len(code15))
            train_code15, val_code15 = Subset(code15, range(0, train_size)), Subset(code15, range(train_size, len(code15)))
            datasets_pretrain.append(train_code15)
            val_datasets.append(val_code15)
        elif dataset == 'code':
            _code = code.ECGCODEDataset(
                config, 
                global_augmentations=get_transforms(config, split='train', type='global'), 
                local_augmentations=get_transforms(config, split='train', type='local')
            )
            # split the dataset into train and val
            train_size = int(0.9 * len(_code))
            train_code, val_code = Subset(_code, range(0, train_size)), Subset(_code, range(train_size, len(_code)))
            datasets_pretrain.append(train_code)
            val_datasets.append(val_code)
        elif dataset == 'ptbxl':
            datasets_pretrain.append(ptb_xl.ECGPTBXLDataset(
                config, 
                split='train', 
                global_augmentations=get_transforms(config, split='train', type='global'), 
                local_augmentations=get_transforms(config, split='train', type='local')
            ))
            val_datasets.append(ptb_xl.ECGPTBXLDataset(
                config, 
                split='val', 
                global_augmentations=get_transforms(config, split='train', type='global'), # I want the training augmentation in this case
                local_augmentations=get_transforms(config, split='train', type='local')
            ))
        elif dataset == 'chapman':
            chapman_dataset = chapman.ECGChapmanDataset(
                config, 
                global_augmentations=get_transforms(config, split='train', type='global'),
                local_augmentations=get_transforms(config, split='train', type='local')
            )
            # split the dataset into train and val
            train_size = int(0.9 * len(chapman_dataset))
            train_chapman, val_chapman = Subset(chapman_dataset, range(0, train_size)), Subset(chapman_dataset, range(train_size, len(chapman_dataset)))
            datasets_pretrain.append(train_chapman)
            val_datasets.append(val_chapman)

        else:
            raise ValueError(f"Dataset {dataset} not found")

    val_dataset = ConcatDataset(val_datasets)
    train_dataset = ConcatDataset(datasets_pretrain)

    return train_dataset, val_dataset


lead_names = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def unpad_signal(signal):
    start_unpad_idx = 0
    while start_unpad_idx < signal.shape[0] and np.all(signal[start_unpad_idx, :] == 0):
        start_unpad_idx += 1

    end_unpad_idx = signal.shape[0]
    while end_unpad_idx > start_unpad_idx and np.all(signal[end_unpad_idx-1, :] == 0):
        end_unpad_idx -= 1

    if start_unpad_idx >= end_unpad_idx:
        return None
    else:
        return signal[start_unpad_idx:end_unpad_idx, :]

def check_sample(record_path):
    # if record exists
    if not os.path.exists(f'{record_path}.hea'):
        print(f"Record {record_path} does not exist - skipping")
        return False
    
    try:
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal
        signal = unpad_signal(signal)
    except Exception as e:
        print(f"Error in record {record_path}: {e}")
        return False

    if signal is None: 
        print(f"Record {record_path} is none - skipping")
        return False
    
    # if hasnan remove
    if np.isnan(signal).any():
        print(f"Record {record_path} has nan - skipping")
        return False
    
    if len(signal) < 360:
        print(f"Record {record_path} has less than 360 samples - skipping")
        return False

    # skip record with too big variance and high values
    if np.var(signal) > 10 and (np.max(signal) >= 15 or np.min(signal) < -15):
        print(f"Record {record_path} has too high variance - skipping")
        return False
    if np.var(signal) < 0.0001:
        print(f"Record {record_path} has too low variance - skipping")
        return False
    
    # check if the signal is empty
    # signal = torch.tensor(signal, dtype=torch.float32)
    # sig_len = (signal != 0.).flip(0).cumsum(dim=0).flip(0).max(dim=-1)[0].max(dim=-1)[0]
    #if sig_len == 0:
    #    print(f"Record {record_path} is empty - skipping")
    #    return False
    #del signal
    
    return True

    if checksums is None:
        x = wfdb.rdrecord(record, physical=False)   
        signals = np.asarray(x.d_signal)
        checksums = np.sum(signals, axis=0, dtype=np.int16)

    header_filename = os.path.join(record + '.hea')
    string = ''
    with open(header_filename, 'r') as f:
        for i, l in enumerate(f):
            if i == 0:
                arrs = l.split(' ')
                num_leads = int(arrs[1])
            if 0 < i <= num_leads and not l.startswith('#'):
                arrs = l.split(' ')
                arrs[6] = str(checksums[i-1])
                l = ' '.join(arrs)
            string += l

    with open(header_filename, 'w') as f:
        f.write(string)