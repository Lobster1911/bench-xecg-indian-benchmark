# Exercise (R-peak detection)

Here is the documentation on how to test your model on Exercise R-peak detection.

Download the dataset from [zenodo](https://zenodo.org/records/5727800) and set `data_folder_high_intensity` in `config_defaults/train_high_intensity_defaults.yaml`.

## Configuration params

This dataset containg 20 second 12 lead ecgs.

### max_length_signal

Define `maximum_len_signal` to be the maximum input lenght of the signal your model accepts.

### plot_predictions

If you want to see the prediction of the model on a signal set `plot_predictions` to `true`: it will automatically upload an image to weights and biases.


## Run the experiment

Run the experiment with `python3 train_r_peak_intense.py --config_file <path_to_your_config>`