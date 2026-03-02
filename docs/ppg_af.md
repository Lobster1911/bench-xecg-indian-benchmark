# PPG Atrial Fibrillation

Here is the documentation on how to test your model on PPG Atrial Fibrillation.

Create a new `yaml` configuration file, this will extend the configuration in `config_defaults/train_deepbeat_defaults.yaml`.

Download the dataset from [synapse](https://www.synapse.org/Synapse:syn21985690/wiki/).
Then set `data_folder_deepbeat` in the new config file with the folder where you extracted the data.

## Configuration params

### data_pct

Because the training set of this dataset comes in a pre-augmented version making this dataset very big (in the order of million of samples). Setting `data_pct` a `WeightedRandomSampler` will load a random subset of the total training set at every epoch, simulating train-time augmentations. We used `data_pct: 0.05` for our experiments.

## Run the experiment

Run the experiment with `uv run run_scripts/train_deepbeat.py --config_file <path_to_your_config>`