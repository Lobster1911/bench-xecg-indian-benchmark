# Age regression

Here is the documentation on how to test your model on Age regression.

Create a new `yaml` configuration file, this will extend the configuration in `config_defaults/train_age_defaults.yaml`.

### CODE15%
Download the CODE15% from [zenodo](https://zenodo.org/records/4916206). This dataset comes in `hdf5` format and you have to extract all the records to `wfdb` format. You can use the `dataset_preparation_utils.save_record_hdf5_to_wfdb` function to extract the signals.
Then set `data_folder_code15` in the new config file with the folder where you extracted the data.

### PTBXL
Download the PTB-XL dataset from [physionet](https://physionet.org/content/ptb-xl/1.0.3/).
Then set `data_folder_ptbxl` in the new config file with the folder where you extracted the data (point to the `records500` folder).

### CPSC
Download the CPSC2018 dataset from [physionet](https://physionet.org/content/challenge-2020/1.0.2/training/cpsc_2018/).
Then set `data_folder_cpsc2018` in the new config file with the folder where you extracted the data.

### MIMIC-IV
Download the MIMIV-IV-ECG dataset from [physionet](https://physionet.org/content/mimic-iv-ecg/1.0/).
Then set `data_folder_mimic` in the new config file with the folder where you extracted the data.

You need to download the diagnostic labels for MIMIC-IV-ECG `records_w_diag_icd10.csv` from [physionet](https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/). This needs crediential access.
An alternative is to get the demographic information from [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/), and again it requires credential access.

## Run the experiment

Run the experiment with `python3 train_age.py --config_file <path_to_your_config>`