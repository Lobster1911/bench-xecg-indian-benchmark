
from pathlib import Path

import torch
import yaml
import pandas as pd
import numpy as np

from dataset.pretraining_dataset import PretrainDataset

class DeepBeatDataset(PretrainDataset):
    def __init__(self, config, split='train', global_augmentations=None, local_augmentations=None):
        super().__init__(config, split=split, global_augmentations=global_augmentations, local_augmentations=local_augmentations)
        self.data_folder = Path(config.data_folder_deepbeat)
        self.signals = None
        self.info_dict = {"fs": 32, "sig_name": ["II"]}

        if split == 'train':
            path = self.data_folder / 'train.npz'
        elif split == 'val':
            path = self.data_folder / 'validate.npz'
        elif split == 'test':
            path = self.data_folder / 'test.npz'
        else:
            raise ValueError(f'Unknown split {split}')

        data = np.load(path, allow_pickle=True)
        self.signals = data['signal']
        self.qa_label = data['qa_label']
        self.rhythm = data['rhythm']
        self.rhythm_label = torch.from_numpy(self.rhythm).float()
        self.parameters = data['parameters']

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        signal = self.resample_if_needed(signal, self.info_dict)
        signal = self.map_leads_and_clean(signal, self.info_dict)

        labels = self.rhythm_label[idx]

        out = {
            'signal': signal.float(),
            'label': labels.float(),
        }
        return out


class ConfigDict(dict):
    def __getitem__(self, key):
        return self.get(key, None)

    def __getattr__(self, key):
        return self.get(key, None)

    def __setattr__(self, key, value):
        self[key] = value

    # merge the two configs, if the key is not in the config file, use the default value
    def update(self, u):
        for k, v in u.items():
            if isinstance(v, dict) and isinstance(self.get(k), dict):
                self[k].update(v)
            else:
                self[k] = v


if __name__ == '__main__':

    base_data_path = Path("../../../data/DeepBeat_PPG")
    val_path = base_data_path / 'validate.npz'

    config_path = "../config_defaults/train_deepbeat_config_defaults.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    config = ConfigDict(config)
    dataset = DeepBeatDataset(config, split="val")

    print(dataset[0])
    print(dataset[0]["signal"].shape)
    print(dataset[1]["label"].shape)
    print("Done!")