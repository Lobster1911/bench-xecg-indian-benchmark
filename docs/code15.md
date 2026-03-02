### CODE15%
Download the CODE15% from [zenodo](https://zenodo.org/records/4916206). 
This dataset comes in `hdf5` format, run the following code to exctract all the data:

```shell
    uv run prepare_dataset.py \
    --data-folder <path-to-code15-dataset> \
    --dataset code15 \
    --label-file-output processed/exams_filtered
```

This script expect the original hdf5 files to be in a subdirectory named `raw`, it will create a new subfolder called `processed` with all the new files, here it is an example:

```shell
path-to-code15-dataset/
├── raw/                         # Put your downloaded .hdf5 files here
│   ├── exams_part0.hdf5
│   ├── exams_part1.hdf5
│   └── ...
└── processed/                   # Created by the script
    ├── exams_filtered.csv       # The label file
    ├── 316508.dat               # WFDB signal file
    ├── 316508.hea               # WFDB header file
    └── ...
```

Once the extraction is complete, update your `config.yaml` to point to the newly created folder:
```yaml
data_folder_code15: "<path-to-code15-dataset>/processed"