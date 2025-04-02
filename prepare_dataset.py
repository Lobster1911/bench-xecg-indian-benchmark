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
from dataset.generic_utils import get_max_n_jobs
pandarallel.initialize(progress_bar=True)

# MIMIC: python prepare_dataset.py --nk_clean --output_folder /media/Volume/data/MIMIC_IV/nkclean_360_12l/ --dataset mimic
# CODE:  python prepare_dataset.py --nk_clean --output_folder /media/Volume/data/CODE15/nkclean_360_12l/ --data_folder /media/Volume/data/CODE15/raw --label_file /media/Volume/data/CODE15/exams.csv --dataset code15
# PTB-XL: python prepare_dataset.py --nk_clean --output_folder /media/Volume/data/PTB-XL/nkclean_360_12l/ --data_folder /media/Volume/data/PTB-XL/ --label_file /media/Volume/data/PTB-XL/ptbxl_database.csv --dataset ptbxl

parser = argparse.ArgumentParser(description='Create dataset for MIT-BIH')
parser.add_argument('--data_folder', type=str, default='/media/Volume/data/MIMIC_IV/', help='Path to raw data folder')
parser.add_argument('--label_file', type=str, default='/media/Volume/data/MIMIC_IV/records_w_diag_icd10.csv', help='Path to the label file')
parser.add_argument('--nk_clean', action='store_true', help='Use NeuroKit2 to clean the data')
parser.add_argument('--output_folder', type=str, required=True, help='the name of the output directory')
parser.add_argument('--dataset', type=str, required=True, help='the name of the dataset: mimic, code 15 or ptbxl')
args = parser.parse_args()

def process_sample_mimic(sample):
    record_path = os.path.join(args.data_folder, 'files', 'p' + str(sample['subject_id'])[:4], 'p' + str(sample['subject_id']), 's' + str(sample['study_id']), str(sample['study_id']))
    out_path = os.path.join(args.output_folder, str(sample['study_id']))
    return resample_and_save_record_wfdb(record_path, 360, out_path, nk_clean=args.nk_clean)

def process_sample_code15(exam):
    record_path = os.path.join(args.data_folder, str(exam[1]['trace_file']))
    exam_id = exam[1]['exam_id']
    out_path = os.path.join(args.output_folder, str(exam[1]['exam_id']))
    return resample_and_save_record_hdf5(record_path, exam_id, 360, out_path, nk_clean=args.nk_clean)

def process_sample_ptbxl(sample):
    record_path = os.path.join(args.data_folder, sample['filename_hr'])
    out_path = os.path.join(args.output_folder, sample['filename_hr'].split('/')[1])
    # print(out_path)
    # out_path = os.path.join(args.output_folder, str(sample['ecg_id']))
    return resample_and_save_record_wfdb(record_path, 360, out_path, nk_clean=args.nk_clean)


def add_labels_mimic(exams):
    exams['is_male'] = exams.parallel_apply(lambda x: x['gender'] == 'M', axis=1)

    exams['icd10_codes_set'] = exams.parallel_apply(lambda row: parse_diagnosis(row), axis=1)
    exams['icd10_codes_descs'] = exams.parallel_apply(lambda row: get_desc(row), axis=1)
    exams['icd10_codes_groups'] = exams.parallel_apply(lambda row: get_group(row), axis=1)

    # set the same categories as in the code 15 dataset
    exams['1dAVB'] = exams.parallel_apply(lambda row: '1dAVB' in get_array(row), axis=1)
    exams['RBBB'] = exams.parallel_apply(lambda row: 'RBBB' in get_array(row), axis=1)
    exams['LBBB'] = exams.parallel_apply(lambda row: 'LBBB' in get_array(row), axis=1)
    exams['SB'] = exams.parallel_apply(lambda row: 'SB' in get_array(row), axis=1)
    exams['ST'] = exams.parallel_apply(lambda row: 'ST' in get_array(row), axis=1)
    exams['AF'] = exams.parallel_apply(lambda row: 'AF' in get_array(row), axis=1)

    # add all the other labels
    exams['Other'] = exams.parallel_apply(lambda row: 'Other' in get_array(row), axis=1)
    exams['Hypertension'] = exams.parallel_apply(lambda row: 'Hypertension' in get_array(row), axis=1)
    exams['Ischaemic disease'] = exams.parallel_apply(lambda row: 'Ischaemic disease' in get_array(row), axis=1)
    exams['Pulmonary Heart'] = exams.parallel_apply(lambda row: 'Pulmonary Heart' in get_array(row), axis=1)
    exams['Cerebrovascular diseases'] = exams.parallel_apply(lambda row: 'Cerebrovascular diseases' in get_array(row), axis=1)
    exams['Arteries diseases'] = exams.parallel_apply(lambda row: 'Arteries diseases' in get_array(row), axis=1)
    exams['Veins diseases'] = exams.parallel_apply(lambda row: 'Veins diseases' in get_array(row), axis=1)
    exams['Hypotension'] = exams.parallel_apply(lambda row: 'Hypotension' in get_array(row), axis=1)
    exams['Heart Failure'] = exams.parallel_apply(lambda row: 'Heart Failure' in get_array(row), axis=1)
    exams['Cardiomiopathy'] = exams.parallel_apply(lambda row: 'Cardiomiopathy' in get_array(row), axis=1)
    exams['Rheumatic disease'] = exams.parallel_apply(lambda row: 'Rheumatic disease' in get_array(row), axis=1)
    return exams

def process_csv_file_mimic(csv_file, out_csv_file, records_to_remove=None):
    """
    Transform the icd10 codes into a ready to use format aligned with common classes

    :param csv_file: the path to the csv file
    :param out_csv_file: the path to the output csv file
    :param records_to_remove: the list of records to remove
    :return:
    """
    exams = pd.read_csv(csv_file)

    records_to_remove = [int(r) for r in records_to_remove]

    print(f'Removing {len(records_to_remove)} records ({records_to_remove[0]}, ...)')
    print(f'Original length: {len(exams)}')
    exams = exams.drop(exams[exams['study_id'].isin(records_to_remove)].index)
    print(f'New length: {len(exams)}')
    print('some study ids:', exams['study_id'].head())
    exams.drop(columns=['ecg_no_within_stay', 'ecg_no_within_stay', 'ecg_taken_in_hosp', 'ecg_taken_in_ed_or_hosp', 'anchor_year', 'anchor_age'], inplace=True)

    # ensure correct format of the identifier column
    exams.parallel_apply(lambda row: str(row['study_id']).split('/')[0], axis=1)
    exams.to_csv(out_csv_file)

def process_csv_file_code15(csv_file, out_csv_file, records_to_remove=None):
    """
    Remove the records from the csv file that are in the records_to_remove list, these records contain useless ECG signals

    :param csv_file: the path to the csv file
    :param out_csv_file: the path to the output csv file
    :param records_to_remove: the list of records to remove
    :return:
    """
    exams = pd.read_csv(csv_file)
    records_to_remove = [int(r) for r in records_to_remove]

    print(f'Removing {len(records_to_remove)} records ({records_to_remove[0]}, ...)')
    print(f'Original length: {len(exams)}')
    exams = exams[~exams['exam_id'].isin(records_to_remove)]
    print(f'New length: {len(exams)}')

    exams.to_csv(out_csv_file)    

def process_csv_file_ptbxl(csv_file, out_csv_file, records_to_remove=None):
    """
    Remove the records from the csv file that are in the records_to_remove list, these records contain useless ECG signals

    :param csv_file: the path to the csv file
    :param out_csv_file: the path to the output csv file
    :param records_to_remove: the list of records to remove
    :return:
    """
    exams = pd.read_csv(csv_file)

    records_to_remove = [int(r.split('_')[0]) for r in records_to_remove]

    print(f'Removing {len(records_to_remove)} records ({records_to_remove[0]}, ...)')
    print(f'Original length: {len(exams)}')
    exams = exams[~exams['ecg_id'].isin(records_to_remove)]
    print(f'New length: {len(exams)}')

    exams.to_csv(out_csv_file)

def process_sample_code15(exam):
    record_path = os.path.join(args.data_folder, str(exam[1]['trace_file']))
    exam_id = exam[1]['exam_id']
    out_path = os.path.join(args.output_folder, str(exam[1]['exam_id']))
    resample_and_save_record_hdf5(record_path, exam_id, 360, out_path, nk_clean=args.nk_clean)


if __name__ == '__main__':
    # make directory for the output
    clean_and_create_directory(args.output_folder)

    if args.dataset == 'mimic':
        records = pd.read_csv(os.path.join(args.data_folder, 'machine_measurements.csv'))
        res = Parallel(n_jobs=get_max_n_jobs())(delayed(process_sample_mimic)(sample) for i, sample in tqdm(records.iterrows()))
        res = [r for r in res if r is not None]
        process_csv_file_mimic(args.label_file, os.path.join(args.output_folder, 'records_w_diag_icd10_labelled.csv'), records_to_remove=res)

    if args.dataset == 'code15':
        exams = pd.read_csv(args.label_file)
        res = Parallel(n_jobs=get_max_n_jobs())(delayed(process_sample_code15)(exam) for exam in tqdm(exams.iterrows()))
        res = [r for r in res if r is not None]
        process_csv_file_code15(args.label_file, os.path.join(args.output_folder, 'exams_labelled.csv'), records_to_remove=res)

    if args.dataset == 'ptbxl':
        exams = pd.read_csv(args.label_file)
        res = Parallel(n_jobs=get_max_n_jobs())(delayed(process_sample_ptbxl)(sample) for i, sample in tqdm(exams.iterrows()))
        res = [r for r in res if r is not None]
        process_csv_file_ptbxl(args.label_file, os.path.join(args.output_folder, 'ptbxl_database.csv'), records_to_remove=res)


