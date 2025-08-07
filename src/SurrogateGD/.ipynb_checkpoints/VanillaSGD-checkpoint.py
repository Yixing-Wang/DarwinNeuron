import torch
import torch.optim as optim
from .SurrogateGD import SGD

class SGDModel(SGD):
    def __init__(self, ModelClass, *model_args, Optimizer=optim.Adam, lr=0.001, device='cpu', spike_grad=None, **model_kwargs):
        """
        SGDModel wraps any model for training via gradient descent.

        Args:
            ModelClass (nn.Module): The model class to instantiate.
            *model_args: Positional arguments for ModelClass.
            Optimizer (torch.optim): Optimizer class (default: Adam).
            lr (float): Learning rate for optimizer.
            device (str): 'cpu' or 'cuda'.
            spike_grad (function, optional): Surrogate gradient function from snntorch.surrogate.
            **model_kwargs: Keyword arguments for ModelClass.
        """
        self.device = device

        # Insert spike_grad into model_kwargs if specified
        if spike_grad is not None:
            model_kwargs['spike_grad'] = spike_grad

        self.model = ModelClass(*model_args, **model_kwargs).to(device)
        self.optimizer = Optimizer(self.model.parameters(), lr=lr)

    def update(self, x, y, loss_fn):
        self.model.train()
        self.optimizer.zero_grad()
        y = y.to(dtype=torch.long)
    
        output = self.model(x)
        
        if isinstance(output, tuple):
            _, mem = output
            output = mem[-1]  # Use last time step's membrane potential for classification
    
        loss = loss_fn(output, y)
        loss.backward()
        self.optimizer.step()

    def get_best_model(self):
        return self.model
