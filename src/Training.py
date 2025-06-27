import os
import torch
import csv
import wandb
from pathlib import Path
from dataclasses import dataclass
from Utilities import spike_to_label, voltage_to_logits


@dataclass
class SNNStats:
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

# ValueError: path must be a file, directory or externalreference like s3://bucket/path
# wandb: ERROR FileNotFoundError: [Errno 2] No such file or directory: 'best-model.pth'
# def log_model(es_model,run): 
#     filename = 'best-model.pth'
#     model = es_model.get_best_model()
#     torch.save(model.state_dict(), filename)
#     run.log_model(path=filename)
#     os.remove(filename)

def log_model(es_model, run, epoch=None):
    filename = f"best-model-epoch{epoch}.pth"
    model = es_model.get_best_model()
    torch.save(model.state_dict(), filename)

    artifact = wandb.Artifact(name=f"best-model-epoch{epoch}.pth", type='model')
    artifact.add_file(filename)
    run.log_artifact(artifact)

    os.remove(filename)


# record data to local csv
def maybe_log_to_csv(entry_dict, config_dict, results_path):
    if results_path is None:
        return
    combined_dict = {**config_dict, **entry_dict}
    write_header = not Path(results_path).exists()

    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=combined_dict.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(combined_dict)

def val_loop_snn(es_model, dataloader, loss_fn, device):
    model = es_model.get_best_model()
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
    print(f"Test Error: \nAccuracy: {(100*test_acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    # Here, result.spike_count_per_neuron has shape [total_data(sum of all batches), neurons]
    return SNNStats(test_loss, correct, len(dataloader.dataset), torch.cat(spike_count_per_neuron, dim=0))

def train_loop_snn(es_model, train_dataloader, val_dataloader, loss_fn, device, run, epoch=None, results_path=None, config_dict=None):
    """ one epoch of training, going through all the batches once
    """    
    for batch, (x, y) in enumerate(train_dataloader):
        x, y = x.to(device), y.to(device)
        def get_model_stats(model):
            stats = run_snn_on_batch(model, x, y, loss_fn)
            return stats

        es_model.update(get_model_stats)

        ## best model metrics as training metrics
        best_model = es_model.get_best_model()
        best_stats = run_snn_on_batch(best_model, x, y, loss_fn)
        print(f"batch {batch}, loss: {best_stats.loss:>7f}, accuracy: {100 * best_stats.get_accuracy():>0.1f}%")

        ## validation loss and accuracy
        val_stats = val_loop_snn(es_model, val_dataloader, loss_fn, device)
        
        ## record keeping
        # train 
        batch_metrics = {'train_loss': best_stats.loss.item(), \
                'train_acc' : best_stats.get_accuracy(), \
                'train_spike_percentage': best_stats.get_spike_percentage(), \
                'train_average_neuron_spikes': best_stats.get_average_neuron_spikes(), \
        # val 
                'val_loss': val_stats.loss.item(), \
                'val_acc': val_stats.get_accuracy(), \
                'val_spike_percentage': val_stats.get_spike_percentage(), \
                'val_average_neuron_spikes': val_stats.get_average_neuron_spikes()}
        if epoch is not None:
            global_step = epoch * len(train_dataloader) + batch
            batch_metrics['epoch'] = epoch
            run.log(batch_metrics, step=global_step)
        else:
            run.log(batch_metrics) 
        # log_model(es_model, run) # currently, I just logged model at the end of epoches

        # log to csv
        entry = {'epoch': epoch, 'batch': batch, **batch_metrics}
        maybe_log_to_csv(entry, config_dict, results_path)
    
    # log for this epoch
    epoch_metrics = {'epoch': epoch, \
            'epoch_val_loss': val_stats.loss.item(), \
            'epoch_val_acc': val_stats.get_accuracy(), \
            'epoch_val_spike_percentage': val_stats.get_spike_percentage(), \
            'epoch_val_average_neuron_spikes': val_stats.get_average_neuron_spikes()}
    if epoch is not None:
        step = (epoch + 1) * len(train_dataloader)
        epoch_metrics['epoch'] = epoch
        run.log(epoch_metrics, step=step)
    else:
        run.log(epoch_metrics)
    # log the best model at the end of the epoch
    log_model(es_model, run, epoch)
    