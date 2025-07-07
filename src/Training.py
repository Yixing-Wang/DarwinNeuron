import os, uuid, torch, csv, wandb
from pathlib import Path

from torch.nn.utils import parameters_to_vector

from src.Utilities import run_snn_on_batch, evaluate_snn
from src.LandscapeAnalysis import get_parameter_to_loss_fn, log_loss_grids, log_loss_plot

def log_model(es_model,run, epoch=None):
    os.makedirs('wandb/tmp/', exist_ok=True)
    filename = f"wandb/tmp/{run.id}-{uuid.uuid4().hex[:8]}.pth"
    model = es_model.get_best_model()
    torch.save(model.state_dict(), filename)
    model_name = f"{run.id}-batch-log.pth" if epoch is None else f"{run.id}-epoch-{epoch}.pth"
    run.log_model(path=filename, name=model_name)
    os.remove(filename)  
    
# record data to local csv
def maybe_log_to_csv(entry_dict, config_dict, results_path):
    combined_dict = {**config_dict, **entry_dict}
    write_header = not Path(results_path).exists()

    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=combined_dict.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(combined_dict)

def train_loop_snn(es_model, train_dataloader, val_dataloader, loss_fn, device, run, epoch, results_path=None, loss_plotter=None):
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
        val_stats = evaluate_snn(es_model.get_best_model(), val_dataloader, loss_fn, device)
        
        ## record keeping
        batch_metrics = {
            'epoch':epoch,\
            # training
            'train_loss': best_stats.loss.item(), \
            'train_acc' : best_stats.get_accuracy(), \
            'train_spike_percentage': best_stats.get_spike_percentage(), \
            'train_average_neuron_spikes': best_stats.get_average_neuron_spikes(), \
            # val 
            'val_loss': val_stats.loss.item(), \
            'val_acc': val_stats.get_accuracy(), \
            'val_spike_percentage': val_stats.get_spike_percentage(), \
            'val_average_neuron_spikes': val_stats.get_average_neuron_spikes()}
        global_step = epoch * len(train_dataloader) + batch + 1
        
        # log to wandb
        run.log(batch_metrics, step=global_step)
        # log model 
        log_model(es_model, run)        
        # log to csv
        if results_path is not None:
            entry = {'run_id': run.id, 'batch': batch, 'step': global_step, **batch_metrics}
            maybe_log_to_csv(entry, dict(run.config), results_path)

    ## log for this epoch
    # log metrics
    run.log({
        'epoch': epoch, \
        'epoch_val_loss': val_stats.loss.item(), \
        'epoch_val_acc': val_stats.get_accuracy(), \
        'epoch_val_spike_percentage': val_stats.get_spike_percentage(), \
        'epoch_val_average_neuron_spikes': val_stats.get_average_neuron_spikes()},
        step = (epoch + 1) * len(train_dataloader))
    
    # log best model
    log_model(es_model, run, epoch)
    
    # log loss surface
    if loss_plotter is not None:
        ILLUMINATE_RESOLUTION = 21
        PLOT_RESOLUTION = 1024
        range_lim = 2 * run.config.std
        
        center = parameters_to_vector(es_model.get_best_model().parameters()).cpu().numpy()
        f = get_parameter_to_loss_fn(val_dataloader, es_model.get_best_model(), loss_fn, device)
        parameter_grid, loss_grid = loss_plotter.illuminate_2d(center, f, range_lim, ILLUMINATE_RESOLUTION)
        fig = loss_plotter.get_plot(PLOT_RESOLUTION)
        
        log_loss_plot(fig, run)
        log_loss_grids(parameter_grid, loss_grid, run)
