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
    def update(self, model_loss_fn):
        """Update estimation of the parameters using the samples on the loss landscape.

        Args:
            model_loss_fn (function): this is the loss landscape function, a function which takes a model and returns 
            its loss. The return value is a scalar tensor.
        """
        pass
    
    @abstractmethod
    def get_best_model(self):
        pass