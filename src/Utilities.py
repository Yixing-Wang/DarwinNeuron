import torch
import numpy as np
import os
from pathlib import Path
import datetime
import csv
import random

def init_result_csv(config, project):
    """
    create CSV file by run.config, load headers
    """
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    filename = f"{project}.csv"
    csv_path = results_dir / filename

    # terms to record
    metric_keys = [
        "epoch", "batch",
        "train_loss", "train_acc", "train_spike_percentage", "train_avg_spikes",
        "val_loss", "val_acc", "val_spike_percentage", "val_avg_spikes"
    ]

    # hyperparameter + metric keys
    config_keys = list(config.keys())
    all_keys = config_keys + metric_keys

    # header
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        if write_header:
            writer.writeheader()

    return str(csv_path), config_keys, metric_keys

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # For CUDA (if used)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # hash deterministic
    os.environ["PYTHONHASHSEED"] = str(seed)

def spike_to_label(spike_train, scheme = 'highest_voltage'):
    """Convert spike train to the label in one-hot encoded class

    Args:
        spike_train (tensor): spike train with shape [time_steps, batch_size, classes]
        scheme(string): options: 'most_spikes' and 'first_spike'
        
    Return:
        one label for each sample: [batch_size,]
    """
    if scheme == 'most_spikes':
        # count number of spikes along the time_steps dimension. Result is [batch_size, classes]
        spike_counts = spike_train.count_nonzero(dim=0)
        
        # pick the index of along the clsses dimension
        result = spike_counts.argmax(dim=-1)
    elif scheme == 'highest_voltage':
        # here the spike train is actually voltages with shape [time steps, batch, classes]
        mem_out_aux = torch.max(spike_train, dim = 0)[0] # 0 is indexing max value (rather than indiceis)
        result = torch.argmax(mem_out_aux, dim=1) # result is of shape [batch size]
    else:
        raise Exception('Undefined Scheme')
    
    return result

def spike_count(spike_train):
    """count number of spikes for each spike train

    Args:
        spike_train (tensor): shape v[time_steps, batch_size, classes]

    Returns:
        tensor: [batch_size, classes]
    """
    return spike_train.count_nonzero(dim=0)

def voltage_to_logits(voltage, scheme = 'highest-voltage'):
    # voltage shape: [time steps, batch, classes]
    if scheme == 'highest-voltage':
        result = voltage.max(dim=0)[0]
    else:
        raise('mechanism not defined')
    
    return result

def spike_train_to_events(spike_train):
    # spike_train shape: [time_steps, neurons]
    # output shape: [events, 2]. ([time step, neuron] pairs)
    
    non_zeros_indicies = np.nonzero(spike_train) # nonzeros: ([...dim 1 indicies for nonzero....], [...dim 2 indicies...])
    return np.stack(non_zeros_indicies).T