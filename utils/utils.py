
from sklearn.metrics import confusion_matrix
from prettytable import PrettyTable  # Import PrettyTable for table formatting
import torch
import numpy as np
import random
from collections import Counter
from models.simple_LSTM import ECG_LSTM, ECG_CONV1D_LSTM
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import yaml
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from torch.utils.data import Subset
import numpy as np
from models.classification import xLSTMClassification
from ecg_jepa.models import load_encoder
import st_mem.encoder as encoder
from ecg_founder.finetune_model import ft_12lead_ECGFounder, ft_1lead_ECGFounder


def get_base_model(config):
    if config.use_st_mem:
        base_model = encoder.__dict__['st_mem_vit_base'](seq_len=2250, patch_size=75, num_leads=12, num_classes=config.num_classes, linear_probing=config.linear_probing, drop_path_rate=config.drop_path_prob)
        checkpoint = torch.load('pretrained_models/st_mem_vit_base_encoder.pth', weights_only=False)
        checkpoint_model = checkpoint['model']
        state_dict = base_model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Remove key {k} from pre-trained checkpoint")
                del checkpoint_model[k]
        msg = base_model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
    elif config.use_ecg_jepa:
        ckpt_dir = 'pretrained_models/multiblock_epoch100.pth'
        base_model = load_encoder(ckpt_dir=ckpt_dir, drop_path_rate=config.drop_path_prob, num_classes=config.num_classes, linear_probing=config.linear_probing) # dim is the dimension of the latent space
    elif config.use_ecg_founder:
        if len(config.leads) == 1:
            path = './checkpoint/1_lead_ECGFounder.pth'
            base_model = ft_1lead_ECGFounder('cuda', path, config.num_classes, linear_prob=config.linear_probing)
        else:
            path = './checkpoint/12_lead_ECGFounder.pth'
            base_model = ft_12lead_ECGFounder('cuda', path, config.num_classes, linear_prob=config.linear_probing)
    else:
        base_model = xLSTMClassification(config=config, num_classes=config.num_classes, num_channels=len(config.leads))
        if config.checkpoint is not None and config.checkpoint != '':   
            checkpoint = torch.load(config.checkpoint, weights_only=False)
            new_state_dict = {format_keys(k): v for k, v in checkpoint['state_dict'].items()}
            # remove the fc layer
            new_state_dict = {k: v for k, v in new_state_dict.items() if 'fc' not in k}
            message = base_model.load_state_dict(new_state_dict, strict=False) 
            print(message) 
    return base_model


def split_dataset_preserve_labels(dataset, split_ratio=0.1, key='class_label'):
    print(f"Splitting dataset with {split_ratio} training data")
    multilabels = np.array([dataset[i][key] for i in range(len(dataset))])  # get multilabels
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=1 - split_ratio)
    train_idx, _ = next(splitter.split(np.zeros(len(multilabels)), multilabels))
    balanced_train_dataset = Subset(dataset, train_idx)
    print(f"Train dataset size: {len(balanced_train_dataset)}")
    return balanced_train_dataset

def format_keys(key):
    if key.startswith('model.'):
        key = key[6:]

    key = key.replace('xlstm.model', 'core.model')  # Remove 'module.' prefix if present
        
    return key

class ConfigDict(dict):
    def __getitem__(self, key):
        return self.get(key, None)

    def __getattr__(self, key):
        return self.get(key, None)

    def __setattr__(self, key, value):
        self[key] = value

    # merge the two configs, if the key is not in the config file, use the default value
    def update(self, u):
        for k, v in u.items():
            if isinstance(v, dict) and isinstance(self.get(k), dict):
                self[k].update(v)
            else:
                self[k] = v

def parse_config(config_file, default_config_file):
    with open(default_config_file, 'r') as file:
        default_config = yaml.safe_load(file)

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    merged_config = ConfigDict(default_config)
    merged_config.update(config)
    # print(merged_config)

    # perform some checks
    if merged_config.use_ecg_jepa:
        merged_config.sampling_freq = 250
        merged_config.patch_size = 50
        # merged_config.max_length_signal = 10
        merged_config.win_len = 1250
        merged_config.leads = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        merged_config.window_size_train = 1000
        merged_config.window_size_val = 1000
    elif merged_config.use_st_mem:
        merged_config.sampling_freq = 250
        merged_config.patch_size = 75
        # merged_config.max_length_signal = 10
        merged_config.win_len = 1125
        merged_config.low_pass_filter = 40
        merged_config.high_pass_filter = 0.67
        merged_config.standardize = True
        merged_config.window_size_train = 1000
        merged_config.window_size_val = 1000
    elif merged_config.use_ecg_founder:
        merged_config.sampling_freq = 500
        merged_config.low_pass_filter = 30
        merged_config.high_pass_filter = 0.5
        merged_config.max_length_signal = 10
        merged_config.window_size_train = 1000
        merged_config.window_size_val = 1000

    if merged_config.linear_probing:
        merged_config.layerwise_lr_decay = 0.
        merged_config.drop_path_prob = False

    merged_config.use_transformers = merged_config.use_ecg_jepa or merged_config.use_st_mem or merged_config.encoder_type == 'transformer'
    return merged_config

def parse_sweep_config(config, default_config_file):
    with open(default_config_file, 'r') as file:
        default_config = yaml.safe_load(file)

    merged_config = ConfigDict(default_config)
    merged_config.update(config)
    # print(merged_config)
    
    return merged_config

def print_metrics_table(sensitivity, ppv, specificity, class_names = [ "N", "S", "V", "F", "Q"] ):
  """Prints a formatted table of per-class metrics."""

  table = PrettyTable()
  table.field_names = ["Class", "Sensitivity", "PPV", "Specificity"]

  for i, class_name in enumerate(class_names):
      table.add_row([class_name, f"{sensitivity[i]:.4f}", f"{ppv[i]:.4f}", f"{specificity[i]:.4f}"])

  print(table)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_training_class_weights(train_dataset, do_not_consider_classes = [], label_key='label'):
  """
  Returns the class weights for the training dataset.
  """
  labels = [sample[label_key] for sample in train_dataset]

  if len(labels[0]) > 1:
    # If labels are multilabel, flatten them
    labels = [label for sublist in labels for label in sublist]

  labels = [label.item() for label in labels if label not in do_not_consider_classes]
  
  # labels = train_dataset.get_labels()
  class_counts = Counter(labels)
  total_samples = len(labels)
  num_classes = len(class_counts)

  class_weights = {cls: total_samples / (num_classes * count) for cls, count in class_counts.items()}
  
  weights = torch.tensor([class_weights[cls] for cls in range(num_classes)], dtype=torch.float32)
  print(f"Class Weights: {weights}")
  return weights

def get_training_class_weights_multilabel(train_dataset, label_key='label'):
    labels = [sample[label_key] for sample in train_dataset]
    classes_count = torch.zeros_like(labels[0])

    for label in labels:
        classes_count += label

    num_classes = classes_count.shape[0]
    total_samples = len(labels)

    class_weights =  total_samples / (classes_count * num_classes)
    print('Class weights: ', class_weights)
    return class_weights

