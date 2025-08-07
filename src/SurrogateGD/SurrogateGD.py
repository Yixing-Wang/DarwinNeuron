from abc import ABC, abstractmethod
from torch.nn import Module

class SGD(ABC):
    """
    Abstract base class for Surrogate Gradient Descent Algorithms.
    """
    @abstractmethod
    def __init__(self, Model: Module, device: str):
        """Initialize the SGD with a model and device."""
        pass

    @abstractmethod
    def update(self, x, y, loss_fn):
        """Perform one SGD update step."""
        pass

    @abstractmethod
    def get_best_model(self):
        """Return the best model so far (usually self.model)."""
        pass
