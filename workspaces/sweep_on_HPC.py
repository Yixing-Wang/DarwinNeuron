import torch
import torch.optim as optim
from torch.nn.functional import cross_entropy
from functools import partial
import wandb
import os, time, random
from filelock import FileLock

from src.RandmanFunctions import RandmanConfig, split_and_load
from src.Models import RandmanSNN, spike_regularized_cross_entropy
from src.EvolutionAlgorithms.EvolutionStrategy import ESModel
from src.EvolutionAlgorithms.PseudoPSO import PPSOModel, PPSOModelWithPooling
from src.SurrogateGD.VanillaSGD import SGDModel
from src.Training import train_loop_snn
from src.Utilities import init_result_csv, set_seed
from src.LandscapeAnalysis.LossSurfacePlotter import LossSurfacePlotter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

ENTITY  = "DarwinNeuron"
PROJECT = "big-sweep-full"
SWEEP_FILE = "sweep_id.txt"
LOCK_FILE = SWEEP_FILE + ".lock"

sweep_config = {
    "method": "grid",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {      
        # Seed:
        "seed": {"values": [1,2,3]},  
        # Dataset:
        "nb_input": {"value": 20},
        "nb_output": {"value": 10},
        "nb_steps": {"value": 50},
        "nb_data_samples": {"value": 1000},
        "dim_manifold": {"value": 2},
        "alpha": {"values": [2.0]},
        # SNN:
        "nb_hidden": {"value": 100},
        "learn_beta": {"values": [True, False]},        
        # Training:
        "std": {"values": [0.1]},
        "epochs": {"value":100},
        "batch_size": {"values": [256, 64]},
        "training_method": {"values":["ES", "PPSO", "SGD"]},
        # Optimization:
        "loss_fn": {"values": ["cross_entropy","spk"]},
        "optimizer": {"value": "Adam"},
        "lr": {"values": [0.01, 0.005]},
        "regularization": {"value": "none"},
        # Evolution Strategy:
        "nb_model_samples": {"values": [100]},
        "mirror": {"values": [True]},
}}

def train_snn(config=None):
    with wandb.init(config=config) as run:
        config = wandb.config

        # choose loss function
        if config.loss_fn == "cross_entropy":
            loss_fn = lambda logits, y, spikes=None: cross_entropy(logits, y)
        elif config.loss_fn == "spk":
            loss_fn = partial(spike_regularized_cross_entropy, lambda_reg=1e-3)

        # update current run_name
        keys = ["training_method", "seed", "learn_beta", "lr", "nb_model_samples","loss_fn","batch_size"]
        sorted_items = [str(getattr(config, k, "NA")) for k in sorted(keys)]
        run_name = "-".join(sorted_items)
        run.name = run_name  
        run.save()

        # setting up local csv recording (optional)
        config_dict = dict(run.config)
        config_dict['run_id'] = run.id
        result_path, _, _ = init_result_csv(config_dict, run.project)

        set_seed(config.seed)

        if config.training_method == "PPSO":
            print("generated PPSO")
            my_model = PPSOModelWithPooling(
                RandmanSNN,
                config.nb_input, 
                config.nb_hidden, 
                config.nb_output, 
                config.learn_beta, 
                sample_size=config.nb_model_samples,
                param_std=config.std,
                lr=config.lr,
                device=device,
                mirror=config.mirror,
                acc_threshold=0.90,
                topk_ratio=0.25
            )
        elif config.training_method == "ES":
            print("generated ES")
            my_model = ESModel(
                RandmanSNN,
                config.nb_input,
                config.nb_hidden,
                config.nb_output,
                config.learn_beta,
                sample_size=config.nb_model_samples,
                param_std=config.std,
                Optimizer=optim.Adam,
                lr=config.lr,
                device=device,
                mirror=config.mirror,
            )
        elif config.training_method == "SGD":
            print("generated SGD")
            my_model = SGDModel(
                RandmanSNN,
                config.nb_input,
                config.nb_hidden,
                config.nb_output,
                config.learn_beta,
                spike_grad=None,   # ← default surrogate gradient
                Optimizer=optim.Adam,
                lr=config.lr,
                device=device,
            )

        # load dataset
        train_loader, val_loader = split_and_load(
            RandmanConfig(
                nb_classes=run.config.nb_output,
                nb_units=run.config.nb_input,
                nb_steps=run.config.nb_steps,
                nb_samples=run.config.nb_data_samples,
                dim_manifold=run.config.dim_manifold,
                alpha=run.config.alpha,
            ).read_dataset(),
            run.config.batch_size,
        )

        # loss surface plotter
        # plotter_dir = f"results/{run.project}/runs/{run.id}/"
        # os.makedirs(plotter_dir, exist_ok=True)
        # loss_plotter = LossSurfacePlotter(plotter_dir+"illuminated_loss_surface.npz")

        # epochs
        for epoch in range(config.epochs):
            print(f"Epoch {epoch}\n-------------------------------")

            # train the model
            train_loop_snn(my_model, train_loader, val_loader, loss_fn, device, run, epoch, result_path, loss_plotter=None)

if __name__ == "__main__":
    time.sleep(random.randint(0, 15))
    # 1) In the first run，create sweep_id and write in sweep_id.txt
    with FileLock(LOCK_FILE):
        if os.path.exists(SWEEP_FILE):
            with open(SWEEP_FILE, "r") as f:
                sweep_id = f.read().strip()
        else:
            sweep_id = wandb.sweep(sweep_config, entity=ENTITY, project=PROJECT)
            with open(SWEEP_FILE, "w") as f:
                f.write(sweep_id)
            print("Created sweep:", sweep_id)

    # 2) initialize agent by function
    # wandb.agent(sweep_id, train_snn)
    wandb.agent(
        sweep_id=sweep_id,
        function=train_snn,
        entity=ENTITY,
        project=PROJECT,
        count=None  # assign jobs to agents till all jobs done
    )
