# Blood test

Here is the documentation on how to test your model on blood test task.

Create a new `yaml` configuration file, this will extend the configuration in `config_defaults/train_blood_test_defaults.yaml`.

### MIMIC-IV
Download the MIMIV-IV-ECG dataset from [physionet](https://physionet.org/content/mimic-iv-ecg/1.0/).
Then set `data_folder_mimic` in the new config file with the folder where you extracted the data.

From [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/) download the `labevents.csv.gz` and `d_labitems.csv.gz` files. You need credential access.

## Run the experiment

Run the experiment with `uv run run_scripts/train_lab_mimic.py --config_file <path_to_your_config>`