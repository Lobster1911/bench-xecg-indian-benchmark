# Survival analysis

Here is the documentation on how to test your model on mortality task.

Create a new `yaml` configuration file, this will extend the configuration in `config_defaults/train_mortality_defaults.yaml`.


### CODE15%
Download the CODE15% from [zenodo](https://zenodo.org/records/4916206). This dataset comes in `hdf5` format and you have to extract all the records to `wfdb` format. You can use the `dataset_preparation_utils.save_record_hdf5_to_wfdb` function to extract the signals.
Then set `data_folder_code15` in the new config file with the folder where you extracted the data.

### MIMIC-IV
Download the MIMIV-IV-ECG dataset from [physionet](https://physionet.org/content/mimic-iv-ecg/1.0/).
Then set `data_folder_mimic` in the new config file with the folder where you extracted the data.
From [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/) download the `patients.csv.gz` and `admissions.csv.gz` files. You need credential access.

## Run the experiment

Run the experiment with `python3 train_mortality.py --config_file <path_to_your_config>`