import torch
import torch.optim as optim
from math import ceil

from .EvolutionAlgorithm import EA

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ESParameter:    
    def __init__(self, para_means, para_std, Optimizer = optim.Adam, lr = 0.01, mirror = True, device = default_device, **kwargs):
        """Constructor of ESParameter

        Args:
            para_means (torch.tensor): initial estimation of parameter. This also determines its shape
            para_std (float): std of normal distribution of params.
            Optimizer (torch.optim, optional): torch optimizer which operates using tensor.loss. Defaults to optim.Adam.
            lr (float, optional): learning rate of optimizer. Defaults to 0.01.
            **kwargs: rest of parameters for the optimizer constructor (such as weight_decay for AdamW)
        """
        assert (type(para_means) == torch.Tensor)
        assert para_means.requires_grad == False
        self.means = para_means.data.clone().detach().to(device)
        self.means.grad = torch.zeros(self.means.shape).to(device)
        self.STD = para_std
        self.samples = None
        self.optimizer = Optimizer([self.means], lr=lr, **kwargs)
        self.mirror = mirror 
        self.device = device
        
    def sample(self, sample_size):
        """draw samples for each parameter from normal distribution with self.means and self.STD.

        Args:
            sample_size (int): number of samples

        Returns:
            tensor: shape [sample_size, ...shape of parameters...]
        """        
        # expand self.means to the shape [sample_size, ...shape of parameters...]
        sample_means = self.means.unsqueeze(0).expand(ceil(sample_size/2) if self.mirror else sample_size, *self.means.shape)
        
        # draw samples
        self.samples = torch.normal(mean=sample_means, std=self.STD)
        
        if self.mirror:
            # mirror the samples, so we have negative samples as well
            self.samples = torch.cat([self.samples, 2*self.means-self.samples], dim=0)
        
        return self.samples
       
    def gradient_descent(self, loss):
        """ Move the means of the parameters against gradient. The gradient is calculated based on loss.
            And self.optimizer will be used to step.

        Args:
            loss (Tensor): with shape [nb_samples,]
        """
        # shape of self.samples = [nb_samples, ...weight shape...]
        
        # Result is the gradients for each prameter, so the shape should match
        # with the parameters, which is [...weight shape...].
        
        # Calculate the gradient for each sample weight, so log_grad will have [nb_samples, ...weight shape...]
        log_grad = (self.samples - self.means) / self.STD**2
        
        ## Calculate the sum of log_grad = [nb_samples, ...weight shape...] and loss = [nb_samples,]
        # Reshape loss for broadcasting to [nb_samples, 1....1]
        new_shape = [loss.shape[0]] + [1] * (len(log_grad.shape) - 1)
        
        # grad is now [nb_samples, ...weight shape...]
        grad = log_grad * loss.reshape(new_shape)
        
        # Take average across sample dimension to estimate the expectation
        self.means.grad = grad.mean(dim=0)
        
        # step the optimizer
        self.optimizer.step()
   
        
class ESModel(EA):
    """
    The class which keeps track of all the parameters in a model, handles sampling and updates
    """
   
    def __init__(self, Model, *model_args, sample_size=20, param_std=0.05, Optimizer = optim.Adam, lr = 0.01, mirror=True, device = default_device, **kwargs):    
        """initialize ESModel with the Model class and the standard deviation of the parameters. The means of the parameters are initialized to be the initial parameters of the model.

        Args:
            Model (nn.Module): the model whose parameters are to be optimized
            param_std (float): standard deviation of the parameters, typically 0.01 to 0.05
            Optimizer (torch.optim, optional): optimizer for ES. Defaults to optim.Adam.
        """
        # self.param_dict has ('param_name', ESParam) pairs, each for one layer of the model
        self.param_dict = {name : ESParameter(para_means = param.clone().detach().to(device), para_std = param_std, 
                                              Optimizer = Optimizer, lr=lr, mirror=mirror, **kwargs) for name, param in Model(*model_args).named_parameters()}         
        self.sample_size = sample_size
        self.Model = Model
        self.model_args = model_args
        self.device = device
        
    def _params_to_model(self, param_list):
        model = self.Model(*self.model_args).to(self.device)
        for i, param in enumerate(model.parameters()):
            param.data.copy_(param_list[i])
        return model
           
    def samples(self):
        # each layer samples from the normal distribution
        for es_param in self.param_dict.values():
            es_param.sample(self.sample_size)
        
        for i in range(self.sample_size):
            param_list = [es_param.samples[i] for es_param in self.param_dict.values()]
            yield(self._params_to_model(param_list))
            
    def gradient_descent(self, loss):
        # each layer updates the means of the parameters
        for es_param in self.param_dict.values():
            es_param.gradient_descent(loss)
            
    def get_best_model(self):
        # get the best model
        param_list = [es_param.means for es_param in self.param_dict.values()]
        return self._params_to_model(param_list)
    
    def update(self, model_loss_fn):
        """update estimation of the parameters using the samples on the loss landscape.

        Args:
            model_loss_fn (method): a function which takes a model and returns the loss for that model. The return
                value is a scalar tensor.
        """
        samples_loss = []
        for model in self.samples():
            samples_loss.append(model_loss_fn(model))            
            
        samples_loss = torch.stack(samples_loss) 
        self.gradient_descent(samples_loss)
    
    @staticmethod
    def estimated_loss(loss):
        """calculate the estimated loss for each parameter in the model

        Args:
            loss (Tensor): with shape [nb_samples,]

        Returns:
            Tensor: with shape [nb_params,]
        """
        return torch.mean(loss)