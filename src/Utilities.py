import torch, os, datetime, csv, random, uuid
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict

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
        "epoch", "batch", "step",
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

@dataclass
class SNNStats:
    """
    SNNStats is a utility class for tracking and analyzing statistics of a Spiking Neural Network (SNN) during training or evaluation.
    Attributes:
        loss (torch.Tensor): The loss value for the current batch or epoch.
        correct (int): The number of correctly classified samples.
        total (int): The total number of samples evaluated.
        spike_count_per_neuron (torch.Tensor): A tensor containing the spike counts for each neuron in the batch. Shape: [batch_size, neurons]
    Methods:
        get_accuracy():
            Calculates and returns the classification accuracy as a float.
        get_spike_percentage():
            Computes the percentage of neuron activations (spikes) above a threshold (0.5) across all samples and neurons.
        get_average_neuron_spikes():
            Returns the average number of spikes per neuron across the batch as a float.
    """
    loss: torch.Tensor
    correct: int
    total: int
    spike_count_per_neuron: torch.Tensor # shape:[batch_size, neurons]
    
    def get_accuracy(self):
        return self.correct / self.total
    
    def get_spike_percentage(self):
        return ((self.spike_count_per_neuron>0.5).sum() / self.spike_count_per_neuron.numel()).item()
    
    def get_average_neuron_spikes(self):
        return self.spike_count_per_neuron.mean().item()

def run_snn_on_batch(model, x, y, loss_fn): 
    # shape: [time_steps, batch_size, classes]
    spikes, voltages = model(x)
    pred_y = spike_to_label(voltages, scheme = 'highest_voltage')
    logits = voltage_to_logits(voltages, scheme='highest-voltage')
    
    loss = loss_fn(logits, y.long())
    correct = (pred_y == y).sum().item()
    spike_count_per_neuron = spikes.sum(dim=0) # dim: [batch_size, neurons]
    stats = SNNStats(loss, correct, len(y), spike_count_per_neuron)
    
    return stats

def evaluate_snn(model, dataloader, loss_fn, device):
    """
    Evaluates a spiking neural network (SNN) model on a given dataset.
    Args:
        model (torch.nn.Module): The SNN model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader providing input data and labels.
        loss_fn (callable): Loss function to compute the loss.
        device (torch.device): Device on which to perform computation (e.g., 'cpu' or 'cuda').
    Returns:
        SNNStats: An object containing the average loss, total number of correct predictions,
                  total number of samples, and concatenated spike counts per neuron for all batches.
    Side Effects:
        Prints the accuracy and average loss over the dataset.
    """    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    spike_count_per_neuron = []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        stats = run_snn_on_batch(model, x, y, loss_fn) 
        test_loss += stats.loss
        correct += stats.correct
        spike_count_per_neuron.append(stats.spike_count_per_neuron) 

    test_loss /= num_batches
    test_acc = correct / size
    print(f"Accuracy: {(100*test_acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    # Here, result.spike_count_per_neuron has shape [total_data(sum of all batches), neurons]
    return SNNStats(test_loss, correct, len(dataloader.dataset), torch.cat(spike_count_per_neuron, dim=0))

def next_id(df):
    return 0 if (df is None) or df.empty else df.index.max() + 1

def match_config(df: pd.DataFrame, config):
    """
    Find DataFrame rows matching all dataclass parameter values.
    
    Args:
        df (pd.DataFrame): DataFrame to search with matching column names
        config: Any dataclass instance with parameters to match
        
    Returns:
        pd.DataFrame: Rows where all parameter values match, empty if none found
    """
    
    param_keys = list(asdict(config).keys())
    match = df.loc[
        (df[param_keys] == pd.Series(asdict(config))).all(axis=1)
    ]
    return match