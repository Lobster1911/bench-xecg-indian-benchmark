import os
import argparse
import pandas as pd
from dataset.dataset_preparation_utils import *
import json
from pandarallel import pandarallel


parser = argparse.ArgumentParser(description='Create dataset for MIT-BIH')
parser.add_argument('--data_folder', type=str, default='/media/Volume/data/MIMIC_IV/', help='Path to raw data folder')
parser.add_argument('--label_file', type=str, default='/media/Volume/data/MIMIC_IV/records_w_diag_icd10.csv', help='Path to the label file')
parser.add_argument('--dataset', type=str, required=True, help='the name of the dataset: mimic, code 15 or ptbxl')
parser.add_argument('--num_workers', type=int, default=16, help='number of parallel jobs')
args = parser.parse_args()

pandarallel.initialize(progress_bar=False, verbose=0, nb_workers=args.num_workers)


def find_records(folder, file_extension='.hea'):
    records = set()
    for root, directories, files in os.walk(folder):
        for file in files:
            extension = os.path.splitext(file)[1]
            if extension == file_extension:
                record = os.path.relpath(os.path.join(root, file), folder)[:-len(file_extension)]
                records.add(record)
    records = sorted(records)
    return records

if __name__ == '__main__':

    print(args)

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
    elif args.dataset == 'code':
        with open(os.path.join(args.data_folder, 'RECORDS.txt'), 'r') as f:
            paths = [line.strip() for line in f.readlines()]
        
        exams = pd.DataFrame()
        exams['file_name'] = paths
    elif args.dataset == 'heedb':
        exams = pd.DataFrame()
        all_records = find_records(args.data_folder, file_extension='.hea')
        print('found records: ', len(all_records), all_records[:10])
        exams['file_name'] = all_records
    else:
        exams = pd.read_csv(args.label_file)

    print(exams.head)
    print(exams.columns)
    print('initial count rows: ', len(exams))


    if args.dataset == 'mimic':
        exams['file_name'] = exams.parallel_apply(lambda row:  str(row['file_name']).replace('mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/', ''), axis=1)
        exams['valid'] = exams.parallel_apply(lambda row: check_sample(os.path.join(args.data_folder, str(row['file_name']))), axis=1)
    elif args.dataset == 'ptbxl':
        exams['valid'] = exams.parallel_apply(lambda row: check_sample(os.path.join(args.data_folder, str(row['filename_hr']))), axis=1)
    elif args.dataset in ['chapman', 'code']:
        exams['valid'] = exams.parallel_apply(lambda row: check_sample(os.path.join(args.data_folder, str(row['file_name']))), axis=1)
    elif args.dataset == 'heedb':
        exams['valid'] = exams.parallel_apply(lambda row: check_sample(os.path.join(args.data_folder, str(row['file_name']))), axis=1)

    to_remove = exams[exams['valid'] == False]
    print('to remove: ', len(to_remove))
    print(to_remove.head())

    exams = exams[exams['valid']]
    exams.drop(columns=['valid'], inplace=True)


    # print(exams.head())
    print('final count rows: ', len(exams))
    exams.to_csv(os.path.join(args.data_folder, 'exams_filtered.csv'))