# PTB-XL

Here you can find the documentation on how to test your model on PTB-XL.

Download the dataset from [physionet](https://physionet.org/content/ptb-xl/1.0.3/) and set `data_folder_ptbxl` in `config_defaults/train_ptb_xl_defaults.yaml`. (point to the `records500` folder)


## Configuration params

This is a standard 10 second 12 leads dataset and thus it do not require any special configuration except from training hyperparameters.

## Run the experiment

Run the experiment with `python3 train_ptb_xl.py --config_file <path_to_your_config>`