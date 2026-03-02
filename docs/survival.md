# Survival analysis

Here is the documentation on how to test your model on mortality task.

Create a new `yaml` configuration file, this will extend the configuration in `config_defaults/train_mortality_defaults.yaml`.

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
### MIMIC-IV
Download the MIMIV-IV-ECG dataset from [physionet](https://physionet.org/content/mimic-iv-ecg/1.0/).
Then set `data_folder_mimic` in the new config file with the folder where you extracted the data.
From [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/) download the `patients.csv.gz` and `admissions.csv.gz` files. You need credential access.

## Run the experiment

Run the experiment with `uv run run_scripts/train_mortality.py --config_file <path_to_your_config>`