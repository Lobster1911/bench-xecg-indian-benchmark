# Sleep Apnea-ECG

Here is the documentation on how to test your model on Sleep Apnea-ECG.

Create a new `yaml` configuration file, this will extend the configuration in `config_defaults/train_sleep_apnea_defaults.yaml`.


Download the dataset from [physionet](https://www.physionet.org/content/apnea-ecg/1.0.0/).
Then set `data_folder_sleep_apnea` in the new config file with the folder where you extracted the data.

## Configuration params

This dataset contains overnight recordings. Annotations are present at minute level. 

### window_size
To divide each long recording into smaller samples use the variable `window_size`, this variable should be set with the number of seconds you want each segment to be long. It can be one of the following values:  5, 10, 20, 30 or 60.

## context_size
This variable sets the second of context each sample is given. Half of the context will be given before and after the signal to classify. Usually this is bigger than 0 only if `window_size` is 60. In particular in our work we used: 0, 120, 240 and 480.

## Run the experiment

Run the experiment with `uv run run_scripts/train_sleep_apnea.py --config_file <path_to_your_config>`