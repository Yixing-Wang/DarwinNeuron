from abc import ABC, abstractmethod
from torch.nn import Module

@abstractmethod
class EA(ABC):
    """
    Abstract base class for Evolutionary Algorithms.
    """
    @abstractmethod
    def __init__(self, Model: Module, device: str):
        """Initialize the Evolution Algorithm with a model and device.

        Args:
            Model (Module): The model to be optimized.
            device (str): The device to run the model on (e.g., 'cpu' or 'cuda').
        """
        pass
        
    @abstractmethod
    def update(self, model_stats_fn):
        """Update estimation of the parameters using the samples on the loss landscape.

        Args:
            model_stats_fn (function): this is the loss landscape function, a function which takes a model and returns
            the training metrics (loss, acc, etc.). The return value is an instance of Training.SNNStats.
        """
        pass
    
    @abstractmethod
    def get_best_model(self):
        pass