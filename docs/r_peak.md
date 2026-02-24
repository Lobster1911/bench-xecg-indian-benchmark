# MIT BIH (classification)

Here is the documentation on how to test your model on MIT-BIH (R-peak detection).

Create a new `yaml` configuration file, this will extend the configuration in `config_defaults/train_mit_bih_defaults.yaml`.

Download the dataset from [physionet](https://www.physionet.org/content/mitdb/1.0.0/).
Then set `data_folder_mit` in the new config file with the folder where you extracted the data.

## Configuration params

In this dataset each recording lasts around 30 minutes. Your model should either have a head that reconstruct each feature to the original patch size or a global head that outputs a number of logits as the original input signal (this last method do not work well).

### win_len
To divide each long recording into smaller samples use the variable `win_len`, note that this represent the size of half of the final sample. So if you want to cut the recording into 10 second segments you need to set `win_len=500` (assuming your model uses a frequency of 100Hz).

Differently from the classification task, here we use only non overlapping segments of the original recording.

### split_val_by_patient

Set `split_val_by_patient` to true if you want the validation set to be divided by patient or false to just shuffle the samples and then random split 80% for training and 20% for valiadtion.

### r_peaks_detection

For this task `r_peaks_detection` should be set to `true`

### plot_predictions

If you want to see the prediction of the model on a signal set `plot_predictions` to `true`: it will automatically upload a plot with the ecg and the r-peaks predictions to weights and biases.

## Run the experiment

Run the experiment with `python3 train_mit_bih.py --config_file <path_to_your_config>`