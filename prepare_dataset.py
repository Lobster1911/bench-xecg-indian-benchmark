import neurokit2 as nk
from tqdm import tqdm
import os
import wfdb
import argparse
import numpy as np
import shutil
import pandas as pd
from collections import Counter
from dataset.dataset_preparation_utils import *
from joblib import Parallel, delayed
import json
import wfdb.processing as wp
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

# CODE:  python prepare_dataset.py --data_folder /media/Volume/data/CODE15/processed/ --label_file /media/Volume/data/CODE15/exams.csv --dataset code15
# PTB-XL: python prepare_dataset.py --data_folder /media/Volume/data/PTB-XL/ --label_file /media/Volume/data/PTB-XL/ptbxl_database.csv --dataset ptbxl
# CPSC: python prepare_dataset.py --dataset cpsc2018 --data_folder /media/Volume/data/CPSC2018/

# MIMIC: python prepare_dataset.py --nk_clean --output_folder /media/Volume/data/MIMIC_IV/nkclean_360_12l/ --dataset mimic
# CODE:  python prepare_dataset.py --nk_clean --data_folder /media/Volume/data/CODE15/raw --label_file /media/Volume/data/CODE15/exams.csv --dataset code15
# PTB-XL: python prepare_dataset.py --nk_clean --output_folder /media/Volume/data/PTB-XL/nkclean_360_12l/ --data_folder /media/Volume/data/PTB-XL/ --label_file /media/Volume/data/PTB-XL/ptbxl_database.csv --dataset ptbxl

# CLUSTER:
# PTB_XL: python prepare_dataset.py --data_folder /home/datasets/ptb-xl/raw/1.0.3/ --label_file /home/datasets/ptb-xl/raw/1.0.3/ptbxl_database.csv --dataset ptbxl --output_folder /home/datasets/ptb-xl/nkclean_360_12l/ --nk_clean
# CODE15: python prepare_dataset.py --data_folder /home/datasets/CODE15/raw/ --label_file /home/datasets/CODE15/raw/exams.csv --dataset code15 --output_folder /home/datasets/CODE15/nkclean_360_12l/ --nk_clean
# MIMIC: python prepare_dataset.py --data_folder /home/datasets/MIMIC_IV/raw/ --label_file /home/datasets/MIMIC_IV/raw/records_w_diag_icd10.csv --dataset mimic --output_folder /home/datasets/MIMIC_IV/nkclean_360_12l/ --nk_clean

parser = argparse.ArgumentParser(description='Create dataset for MIT-BIH')
parser.add_argument('--data_folder', type=str, default='/media/Volume/data/MIMIC_IV/', help='Path to raw data folder')
parser.add_argument('--label_file', type=str, default='/media/Volume/data/MIMIC_IV/records_w_diag_icd10.csv', help='Path to the label file')
parser.add_argument('--dataset', type=str, required=True, help='the name of the dataset: mimic, code 15 or ptbxl')
args = parser.parse_args()

def extract_diagnosis_code(file_name):
    record = wfdb.rdheader(file_name)
    for comment in record.comments:
        if comment.startswith('Dx:'):
            return comment.split(': ')[1]

if __name__ == '__main__':

    if args.dataset == 'chapman':
        # open the file RECORDS and read the lines
        with open(os.path.join(args.data_folder, 'RECORDS'), 'r') as f:
            paths = f.readlines()
        
        exams = pd.DataFrame()
        all_files = []
        for path in paths:
            with open(os.path.join(args.data_folder, path.strip(), 'RECORDS'), 'r') as f:
                lines = f.readlines()
            all_files += [os.path.join(path.strip(), line.strip()) for line in lines]

        exams['file_name'] = all_files
    elif args.dataset == 'cpsc2018':
        # list only the directory names in the data folder
        all_dirs = [d for d in os.listdir(args.data_folder) if os.path.isdir(os.path.join(args.data_folder, d))]
        exams = pd.DataFrame()
        all_files = []
        for d in all_dirs:
            with open(os.path.join(args.data_folder, d, 'RECORDS'), 'r') as f:
                lines = f.readlines()
            all_files += [os.path.join(d, line.strip()) for line in lines]
        exams['file_name'] = all_files
    elif args.dataset == 'code':
        with open(os.path.join(args.data_folder, 'RECORDS.txt'), 'r') as f:
            paths = [line.strip() for line in f.readlines()]
        
        exams = pd.DataFrame()
        exams['file_name'] = paths
    else:
        exams = pd.read_csv(args.label_file)

    print(exams.head)
    print(exams.columns)
    print('initial count rows: ', len(exams))

    if args.dataset == 'mimic':
        exams['file_name'] = exams.parallel_apply(lambda row:  str(row['file_name']).replace('mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/', ''), axis=1)
        exams['valid'] = exams.parallel_apply(lambda row: check_sample(os.path.join(args.data_folder, str(row['file_name']))), axis=1)
    elif args.dataset == 'code15':
        exams['valid'] = exams.parallel_apply(lambda row: check_sample(os.path.join(args.data_folder, str(row['exam_id']))), axis=1)
    elif args.dataset == 'ptbxl':
        exams['valid'] = exams.parallel_apply(lambda row: check_sample(os.path.join(args.data_folder, str(row['filename_hr']))), axis=1)
    elif args.dataset in ['chapman', 'cpsc2018', 'code']:
        exams['valid'] = exams.parallel_apply(lambda row: check_sample(os.path.join(args.data_folder, str(row['file_name']))), axis=1)

    to_remove = exams[exams['valid'] == False]
    print('to remove: ', len(to_remove))
    print(to_remove.head())

    exams = exams[exams['valid']]
    exams.drop(columns=['valid'], inplace=True)


    # get the labels from the label file on cspc2018
    if args.dataset == 'cpsc2018':
        exams['diagnosis_code'] = exams.parallel_apply(lambda row: extract_diagnosis_code(os.path.join(args.data_folder,row['file_name'])), axis=1)


    # print(exams.head())
    print('final count rows: ', len(exams))
    exams.to_csv(os.path.join(args.data_folder, 'exams_filtered.csv'))
