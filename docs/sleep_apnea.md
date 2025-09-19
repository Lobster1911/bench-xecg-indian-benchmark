# Sleep Apnea-ECG

Here is the documentation on how to test your model on Sleep Apnea-ECG.

Download the dataset from [physionet](https://www.physionet.org/content/apnea-ecg/1.0.0/) and set `data_folder_sleep_apnea` in `config_defaults/train_sleep_apnea_defaults.yaml`.

## Configuration params

This dataset contains overnight recordings. Annotations are present at minute level. 

### window_size
To divide each long recording into smaller samples use the variable `window_size`, this variable should be set with the number of seconds you want to use. There are 3 scenarios:
- `window_size<60`: in this case `window_size` should be able to divide one minute without rest (e.g. you can use 5, 10, 20, 30). And because the minute level annotation is divided, each new segment will have the same annotation of the original one minute part. For metric calculation sub-segments are aggregated and averaged for each minute segment.
- `window_size=60`: in this case classification is straightforward and the model can have a single head to classify that segment.
- `window_size>60`: here `window_size` has to be a multiple of 60 and the model should aggregate the features to have a prediction for every 60 second in the signal.

### split_val_by_patient

Set `split_val_by_patient` to true if you want the validation set to be divided by patient or false to just shuffle the samples and then random split 80% for training and 20% for valiadtion.

### max_length_signal

This variable is automatically fixed to handle the correct signal lenght: `config.window_size * config.sampling_freq`

## Run the experiment

Run the experiment with `python3 train_sleep_apnea.py --config_file <path_to_your_config>`