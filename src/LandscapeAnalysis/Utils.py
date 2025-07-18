import torch
from copy import deepcopy
from torch.nn.utils import vector_to_parameters

from src.Utilities import evaluate_snn


def get_parameter_to_loss_fn(loader, model, loss, device="cuda"):
    """
    Returns a function that computes the loss for a given parameter vector on the provided model and data loader.

    Args:
        loader: DataLoader for validation or evaluation.
        model: PyTorch model whose parameters will be set.
        loss: Loss function to evaluate (e.g., cross_entropy).
        device (str): Device to run computations on (default: "cuda").

    Returns:
        function: A function that takes a parameter vector and returns the computed loss.
    """
    model = deepcopy(model)

    def parameter_to_loss_fn(vector):
        vector = torch.tensor(vector, dtype=torch.float32, device=device)
        vector_to_parameters(vector, model.parameters())
        stats = evaluate_snn(model, loader, loss, device)
        return stats.loss.item()

    return parameter_to_loss_fn
