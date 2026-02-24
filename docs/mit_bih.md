# MIT BIH (classification)

Here is the documentation on how to test your model on MIT-BIH (classification).

Create a new `yaml` configuration file, this will extend the configuration in `config_defaults/train_mit_bih_defaults.yaml`.

Download the dataset from [physionet](https://www.physionet.org/content/mitdb/1.0.0/).
Then set `data_folder_mit` in the new config file with the folder where you extracted the data.

## Configuration params

In this dataset each recording lasts around 30 minutes. Annotations are present on each heartbeat. For this task your model should be able to perform feature classification with a granularity at least of 250ms (so for each ouput feature we are sure only one heartbeat is present).

### single_hb
Set `single_hb` to `true` if you want the samples to be extracted from the original signal considering a window of 400 timepoints around each heartbeat (400 timepoints considering the dataset frequency - 360Hz - so around 1,1 seconds). This follows some prior works in the literature. This approach is needed for models that cannot do feature classification.

### win_len
To divide each long recording into smaller samples use the variable `win_len`, note that this represent the size of half of the final sample. So if you want to cut the recording into 10 second segments you need to set `win_len=500` (assuming your model uses a frequency of 100Hz).

### split_val_by_patient

Set `split_val_by_patient` to true if you want the validation set to be divided by patient or false to just shuffle the samples and then random split 80% for training and 20% for valiadtion. This variable should remain at `true`.

### skip_majority_class_samples

Most of the heartbeat are of type N (normal) and thus with `skip_majority_class_samples` set to true we skip - only in training phase - 10 consecutive samples if they are all belonging to the Normal class.

### r_peaks_detection

For this task `r_peaks_detection` should be set to `false`

## Run the experiment

Run the experiment with `python3 train_mit_bih.py --config_file <path_to_your_config>`