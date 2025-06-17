import os
import torch
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

def log_model(es_model,run):
    filename = 'best-model.pth'
    model = es_model.get_best_model()
    torch.save(model.state_dict(), filename)
    run.log_model(path=filename)
    os.remove(filename)  

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

def train_loop_snn(es_model, train_dataloader, val_dataloader, loss_fn, device, run):
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
        run.log({'train_loss': best_stats.loss.item(), \
                'train_acc' : best_stats.get_accuracy(), \
                'train_spike_percentage': best_stats.get_spike_percentage(), \
                'train_average_neuron_spikes': best_stats.get_average_neuron_spikes(), \
        # val 
                'val_loss': val_stats.loss.item(), \
                'val_acc': val_stats.get_accuracy(), \
                'val_spike_percentage': val_stats.get_spike_percentage(), \
                'val_average_neuron_spikes': val_stats.get_average_neuron_spikes()}) 
        log_model(es_model, run)