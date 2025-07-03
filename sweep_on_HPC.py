import torch
import torch.optim as optim
from torch.nn.functional import cross_entropy
import wandb
import os

from src.RandmanFunctions import read_randman10_dataset
from src.Models import RandmanSNN
from src.EvolutionAlgorithms.EvolutionStrategy import ESModel
from src.EvolutionAlgorithms.PseudoPSO import PPSOModel, PPSOModelWithPooling
from src.Training import train_loop_snn
from src.Utilities import init_result_csv, set_seed

device = 'cuda' if torch.cuda.is_available() else 'cpu'

ENTITY  = "DarwinNeuron"
PROJECT = "PPSO-Randman10-debug-HPC"
SWEEP_FILE = "sweep_id.txt"

sweep_config = {
    "method": "grid",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {      
        # Seed:
        "seed": {"values": [0, 1, 2]},  
        # Dataset:
        "nb_input": {"value": 100},
        "nb_output": {"value": 10},
        "nb_steps": {"value": 50},
        "nb_data_samples": {"value": 1000},
        # SNN:
        "nb_hidden": {"value": 10},
        "learn_beta": {"value": False},        
        # Training:
        "std": {"values": [0.05, 0.1]},
        "epochs": {"value":30},
        "batch_size": {"values": [256]},
        # Optimization:
        "loss_fn": {"value": "cross-entropy"},
        "optimizer": {"value": "Adam"},
        "lr": {"values": [0.01]},
        "regularization": {"value": "none"},
        # Evolution Strategy:
        "nb_model_samples": {"values": [20, 100]},
        "mirror": {"values": [False, True]},
}}

def train_snn(config=None):
    with torch.no_grad(), wandb.init(config=config) as run:
        config = wandb.config

        keys = ["seed", "std", "batch_size", "lr", "nb_model_samples"]
        sorted_items = [f"{k}{getattr(config, k)}" for k in sorted(keys)]
        run_name = "-".join(sorted_items)
        run.name = run_name  # update current run_name
        run.save()

        # setting up local csv recording (optional)
        config_dict = dict(config)
        result_path, config_keys, metric_keys = init_result_csv(config_dict, run.project)

        # initialize Evolution Strategy instance
        # es_model = ESModel(
        #     RandmanSNN,
        #     config.nb_input,
        #     config.nb_hidden,
        #     config.nb_output,
        #     0.95,
        #     sample_size=config.nb_model_samples,
        #     param_std=config.std,
        #     Optimizer=optim.Adam,
        #     lr=config.lr,
        #     device=device,
        #     mirror=config.mirror,
        # )

        set_seed(config.seed)
        es_model = PPSOModelWithPooling(
            RandmanSNN,
            config.nb_input, config.nb_hidden, config.nb_output, 0.95,
            sample_size=config.nb_model_samples,
            param_std=config.std,
            lr=config.lr,
            device=device,
            mirror=config.mirror,
            acc_threshold=0.90,
            topk_ratio=0.25
        )

        # load dataset
        train_loader, val_loader = read_randman10_dataset(
            "data/randman_10_dataset.pt", batch_size=config.batch_size
        )

        # epochs
        for epoch in range(config.epochs):
            print(f"Epoch {epoch}\n-------------------------------")

            # train the model
            train_loop_snn(
            es_model, train_loader, val_loader, cross_entropy, device, run, epoch, results_path=result_path, config_dict=dict(config)
            )

if __name__ == "__main__":
    # 1) In the first run，create sweep_id and write in sweep_id.txt
    if os.path.exists(SWEEP_FILE):
        sweep_id = open(SWEEP_FILE).read().strip()
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
