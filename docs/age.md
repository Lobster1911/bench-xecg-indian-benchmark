# Age regression

Here is the documentation on how to test your model on Age regression.

### CODE15%
Download the CODE15% from [zenodo](https://zenodo.org/records/4916206). This dataset comes in `hdf5` format and you have to extract all the records to `wfdb` format. You can use the `dataset_preparation_utils.save_record_hdf5_to_wfdb` function to extract the signals.
Then set `data_folder_code15` in `config_defaults/train_age_defaults.yaml`.

### PTBXL
Download the PTB-XL dataset from [physionet](https://physionet.org/content/ptb-xl/1.0.3/) and set `data_folder_ptbxl` in `config_defaults/train_age_defaults.yaml`. (point to the `records500` folder)

### CPSC
Download the CPSC2018 dataset from [physionet](https://physionet.org/content/challenge-2020/1.0.2/training/cpsc_2018/) and set `data_folder_cpsc2018` in `config_defaults/train_age_defaults.yaml`.

### MIMIC-IV
Download the MIMIV-IV-ECG dataset from [physionet](https://physionet.org/content/mimic-iv-ecg/1.0/) and set `data_folder_mimic` in `config_defaults/train_age_defaults.yaml`.

You need to download the diagnostic labels for MIMIC-IV-ECG `records_w_diag_icd10.csv` from [physionet](https://physionet.org/content/mimic-iv-ecg-ext-icd-labels/1.0.1/). This needs crediential access.
An alternative is to get the demographic information from [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/), but again it requires credential access.

## Configuration params

### max_length_signal

To truncate long signals, set `max_length_signal`. After resampling to the target frequency, the pipeline will automatically cut those exceeding the specified number of timepoints.

## Run the experiment

Run the experiment with `python3 train_age.py --config_file <path_to_your_config>`