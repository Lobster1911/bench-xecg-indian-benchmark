# Exercise (R-peak detection)

Here is the documentation on how to test your model on Exercise R-peak detection.

Create a new `yaml` configuration file, this will extend the configuration in `config_defaults/train_high_intensity_defaults.yaml`.

Download the dataset from [zenodo](https://zenodo.org/records/5727800).
Then set `data_folder_high_intensity` in the new config file with the folder where you extracted the data.

## Configuration params

This dataset containg 20 second 12 lead ecgs.

### max_length_signal

Define `maximum_len_signal` to be the maximum input lenght of the signal your model accepts. If `maximum_len_signal` is smaller than the lenght of the signal, the signal will be divided in sub-segments. E.g. if `maximum_len_signal` corresponds to 10 seconds every ECG will be splitted in two samples.

### plot_predictions

If you want to visualize the prediction of the model on a signal set `plot_predictions` to `true`: it will automatically upload an image with the predictions to weights and biases.

## Run the experiment

Run the experiment with `python3 train_r_peak_intense.py --config_file <path_to_your_config>`