import os

import wfdb
import numpy as np
import h5py

from torch.utils.data import Subset, ConcatDataset

from .generic_utils import get_transforms
from .code_dataset import ECGCODE15Dataset, ECGCODEDataset
from .ptb_xl import ECGPTBXLDataset
from .mimic_iv import ECGMIMICDataset
from .chapman import ECGChapmanDataset
from .incart import ECGIncartDataset
from .heedb import ECGHEEDBDataset

def load_datasets(config):

    datasets_pretrain = []
    val_datasets = []

    for dataset in config.pretrain_datasets:
        if dataset == 'heedb':
            _heedb = ECGHEEDBDataset(
                config, 
                global_augmentations=get_transforms(config, split='train', type='global'), 
                local_augmentations=get_transforms(config, split='train', type='local')
            )
            # split the dataset into train and val
            train_size = int(0.9 * len(_heedb))
            train_heedb, val_heedb = Subset(_heedb, range(0, train_size)), Subset(_heedb, range(train_size, len(_heedb)))
            datasets_pretrain.append(train_heedb)
            val_datasets.append(val_heedb)
        elif dataset == 'mimic':
            datasets_pretrain.append(ECGMIMICDataset(
                config, 
                split='train', 
                global_augmentations=get_transforms(config, split='train', type='global'), 
                local_augmentations=get_transforms(config, split='train', type='local')
            ))
            val_datasets.append(ECGMIMICDataset(
                config, 
                split='val', 
                global_augmentations=get_transforms(config, split='train', type='global'), # I want the training augmentation in this case
                local_augmentations=get_transforms(config, split='train', type='local')
            ))
        elif dataset == 'incart':
            _incart = ECGIncartDataset(
                config, 
                split='train', 
                global_augmentations=get_transforms(config, split='train', type='global'), 
                local_augmentations=get_transforms(config, split='train', type='local')
            )
            train_size = int(0.9 * len(_incart))
            train_incart, val_incart = Subset(_incart, range(0, train_size)), Subset(_incart, range(train_size, len(_incart)))
            datasets_pretrain.append(train_incart)
            val_datasets.append(val_incart)
            
        elif dataset == 'code15':
            code15 = ECGCODE15Dataset(
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
            _code = ECGCODEDataset(
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
            datasets_pretrain.append(ECGPTBXLDataset(
                config, 
                split='train', 
                global_augmentations=get_transforms(config, split='train', type='global'), 
                local_augmentations=get_transforms(config, split='train', type='local')
            ))
            val_datasets.append(ECGPTBXLDataset(
                config, 
                split='val', 
                global_augmentations=get_transforms(config, split='train', type='global'), # I want the training augmentation in this case
                local_augmentations=get_transforms(config, split='train', type='local')
            ))
        elif dataset == 'chapman':
            chapman_dataset = ECGChapmanDataset(
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

def check_sample(record_path, check_less_10_seconds=False, check_variance=False):
    # if record exists
    if not os.path.exists(f'{record_path}.hea'):
        print(f"Record {record_path} does not exist - skipping")
        return False
    
    try:
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal
        # signal = unpad_signal(signal)
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
    
    if check_less_10_seconds:
        # get fs and calculate duration
        len_seconds = record.sig_len / record.fs
        reduced_lead = np.concatenate((signal[:, :2], signal[:, 6:]), axis=1)
        first_15_samples_zero = reduced_lead[:15, :].sum()
        last_15_samples_zero = reduced_lead[-15:, :].sum()

        if len_seconds < 10 or first_15_samples_zero == 0 or last_15_samples_zero == 0:
            print(f'Record {record_path} is too short ({len_seconds}s)')
            return False
    
    if len(signal) < 360:
        print(f"Record {record_path} has less than 360 samples - skipping")
        return False
    
    if check_variance:
        # skip record with too big variance and high values
        if np.var(signal) > 10 and (np.max(signal) >= 15 or np.min(signal) < -15):
            print(f"Record {record_path} has too high variance - skipping")
            return False
        if np.var(signal) < 0.0001:
            print(f"Record {record_path} has too low variance - skipping")
            return False
        
    return True


def save_record_hdf5_to_wfdb(record_path, exam_id, output_file_path):
    """
    Save a record from hdf5 to wfdb format
    
    Args:
        record_path (str): path to the hdf5 file
        exam_id (str): exam id to save, from exams.csv
        output_file_path (str): path to save the wfdb file
    """
    with h5py.File(record_path, 'r') as f:
        # idx of the exam_id
        signal_idx = np.where(f['exam_id'][:] == exam_id)[0][0]
        signal = np.array(f['tracings'][signal_idx], dtype=np.float32)
        
        if isinstance(signal, str) or isinstance(signal, int):
            return str(signal)
        
        try:
            wfdb.wrsamp(
                str(exam_id),
                fs=400,
                units=['mV']*12, 
                sig_name=lead_names, 
                p_signal=signal, 
                fmt=['16']*len(lead_names), 
                adc_gain=[1000]*12, 
                baseline=[0]*12,
                write_dir=output_file_path, # output here
            )
        except Exception as e:
            print(f"Error in record {exam_id}: {e}")