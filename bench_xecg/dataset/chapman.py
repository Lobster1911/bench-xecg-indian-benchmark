import pandas as pd

from .pretraining_dataset import PretrainDataset


class ECGChapmanDataset(PretrainDataset):

    def __init__(self, config, split='train', global_augmentations=None, local_augmentations=None):
        super().__init__(config, split=split, global_augmentations=global_augmentations, local_augmentations=local_augmentations)
        self.data_folder = config.data_folder_chapman
        self.labels_file = config.labels_file_chapman
        self.load_tabular_data()
        self.load_records(split)
        
    def load_records(self, split):
        # fold 19 is for testing, while fold 18 is for validation
        self.records = self.tab_data['file_name'].tolist()

    def load_tabular_data(self):
        # get the csv file with the tabular data
        self.tab_data = pd.read_csv(self.labels_file)
        print("Chapman&Ningbo - tabular data: ", self.tab_data.head())
        print(f'Chapman&Ningbo - colums {self.tab_data.columns}')
        print(f'Chapman&Ningbo - number of samples: {len(self.tab_data)}')

    def __len__(self):
        return len(self.records)
         