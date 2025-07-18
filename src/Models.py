import torch
import snntorch as snn
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.nn.utils import parameters_to_vector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RandmanSNN(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, learn_beta, beta=0.95):
        super(RandmanSNN, self).__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden, bias=False)
        self.lif1 = snn.Leaky(beta=beta, learn_beta=learn_beta)

        self.fc2 = nn.Linear(num_hidden, num_outputs, bias= False)
        self.lif2 = snn.Leaky(beta=beta, learn_beta=learn_beta, reset_mechanism='none')

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.zeros_(m.bias)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        batch_size, time_steps, num_neurons = x.shape
        x = x.permute(1, 0, 2)  # (time, batch, neurons)

        mem1, mem2 = [torch.zeros(batch_size, layer.out_features, device=x.device)
                      for layer in [self.fc1, self.fc2]]

        spk1_rec = []
        mem2_rec = []

        for t in range(time_steps):
            spk1, mem1 = self.lif1(self.fc1(x[t]), mem1)
            _, mem2 = self.lif2(self.fc2(spk1), mem2)
            
            spk1_rec.append(spk1)
            mem2_rec.append(mem2)

        # Each stacked item has shape [time_steps, batch_size, num_neurons]
        return torch.stack(spk1_rec, dim=0), torch.stack(mem2_rec, dim=0)  
    
def calc_randmansnn_parameters(nb_inputs, nb_hidden, nb_outputs):
    model = RandmanSNN(num_inputs=nb_inputs, num_hidden=nb_hidden, num_outputs=nb_outputs, learn_beta=False).to('cpu')
    vector = parameters_to_vector(model.parameters())
    return vector.size(0)

def regularized_cross_entropy(pred, y):
    # pred shape: [batch, classes (logits)]
    regularization_term = torch.sigmoid(-15 * torch.abs(pred[:,0] - pred[:, 1]))
    
    return cross_entropy(pred, y) + regularization_term.mean()

## The Compeition Model
# TODO: need to implement self.update() to keep up with the current pipline.
class ExcitationPopulation(nn.Module):
    def __init__(self, nb_neurons, nb_input, nb_inh, beta, nb_decision_steps):
        super().__init__()        
        self.input_fc = nn.Linear(nb_input, nb_neurons, bias = False)
        self.recurrent_fc = nn.Linear(nb_neurons, nb_neurons, bias = False)
        self.inhibition_fc = nn.Linear(nb_inh, nb_neurons, bias = False)
        self.lif = snn.Leaky(beta, learn_beta = False, threshold=1, reset_mechanism='subtract')
        self.readout_fc = nn.Linear(nb_neurons, 1, bias=False)
        self.readout_lif = snn.Leaky(beta=0.95, reset_mechanism = 'none')
        self.nb_decision_steps = nb_decision_steps
        self.init_states(nb_decision_steps)       

    def get_nb_neurons(self):
        return self.input_fc.out_features
    
    def init_states(self, nb_decision_steps=None):
        # excitatory neurons
        self.mem = self.lif.init_leaky().to(device)
        self.last_spks_queue = [torch.zeros(self.get_nb_neurons(), device=device) for _ in range(nb_decision_steps if nb_decision_steps != None else len(self.last_spks_queue))]  
        
        # readout neuron
        self.readout_mem = self.readout_lif.init_leaky().to(device)
        self.readout_mem_rec = []

    def forward(self, input, inhibition):
        # excitatory neurons
        # TODO: 
        # PROBLEM: THE ABS() IS APPLIED IN THE WRONG PLACE. SHOULD APPLY ON WEIGHTS BEFORE LINEAR COMBINATION
        curr = torch.abs(self.input_fc(input)) + torch.abs(self.recurrent_fc(self.last_spks_queue[-1])) - torch.abs(self.inhibition_fc(inhibition))
        spk, self.mem = self.lif(curr, self.mem)
        
        # readout neuron
        readout_curr = torch.abs(self.readout_fc(spk))
        _, self.readout_mem = self.readout_lif(readout_curr, self.readout_mem)    
        
        # update spk record
        self.last_spks_queue.pop(0)
        self.last_spks_queue.append(spk.clone().detach())
        
        # update readout record
        self.readout_mem_rec.append(self.readout_mem.squeeze(dim=-1).clone())
        
        return spk
        
    def get_last_spikes_means(self):
        # stacked shape: [nb_decision_steps, batch_size, nb_neurons]
        # For each batch, the mean should include all the final steps and all the neurons (first and last dimension)
        # return shape: [batch_size,]
        return torch.stack(self.last_spks_queue).mean(dim = [0, 2])
    
    def get_readout(self):    
        # stacked shape [nb_decision_steps, batch_size]
        return torch.stack(self.readout_mem_rec[-self.nb_decision_steps: ]).mean(dim=0)
    
# def test_ep():
#     ep = ExcitationPopulation(nb_neurons=3, nb_input=10, nb_inh=1, beta=0.95, nb_decision_steps=5).to(device)
#     for _ in range(100):
#         fake_spk = torch.rand([64, 10], device=device)
#         fake_inh = torch.rand([64, 1], device=device)
#         out = ep(fake_spk, fake_inh)
#     print(ep.get_readout().shape)
        
# test_ep()

class CompetitionModel(nn.Module):
    def __init__(self, nb_input, nb_ext, nb_inh, beta_ext, beta_inh, nb_decision_steps):
        super().__init__()
        
        # excitatory
        self.excitatory_1 = ExcitationPopulation(nb_ext, nb_input, nb_inh, beta_ext, nb_decision_steps)
        self.excitatory_2 = ExcitationPopulation(nb_ext, nb_input, nb_inh, beta_ext, nb_decision_steps)
        
        # inhibitory.
        self.inh_fc = nn.Linear(nb_ext, 1, bias = False) # Note: two ext share same inh weights
        self.inh_lif = snn.Leaky(beta_inh, learn_beta = False)
        
        # records
        self.nb_decision_steps = nb_decision_steps
        
    def get_nb_ext(self):
        return self.excitatory_1.get_nb_neurons()
    
    def get_nb_inh(self):
        return self.inh_fc.out_features
    
    def init_states(self):
        self.excitatory_1.init_states()
        self.excitatory_2.init_states()
        self.mem_inh = self.inh_lif.init_leaky()
        
    def forward(self, x):        
        # change x shape from [batch, time steps, nb_input] to [time steps, batch, nb_input]
        x = x.permute([1, 0, 2])
        
        # pad time steps for model to go to steady states
        x = torch.cat([x, torch.zeros(5 + self.nb_decision_steps, x.shape[1], x.shape[2], device=device)])
        
        # initalize membrane potentials
        self.init_states()
        
        # init spikes with shape [nb_neurons]. The batch size will be broadcasted
        inh_spk = torch.zeros([self.get_nb_inh()], device=device)
        
        for t in range(len(x)):          
            # excitation
            ext_1_spk = self.excitatory_1(x[t], inh_spk)
            ext_2_spk = self.excitatory_2(x[t], inh_spk)
            
            # inhibition. Inhibitory neurons are excited, so curr should be positive
            curr_inh = torch.abs(self.inh_fc(ext_1_spk)) + torch.abs(self.inh_fc(ext_2_spk))
            
            inh_spk, self.mem_inh = self.inh_lif(curr_inh, self.mem_inh)
        
        # return shape: [batch_size, 2], where column 0 is ext1, column 1 is ext2
        return torch.stack([self.excitatory_1.get_readout(), self.excitatory_2.get_readout()], dim = 1)
    
    def get_mem_rec(self):
        # stacked shape: [batch, time_steps]
        mem_rec_1 = torch.stack(self.excitatory_1.readout_mem_rec, dim = 1)
        mem_rec_2 = torch.stack(self.excitatory_2.readout_mem_rec, dim = 1)
        
        # return shape: [batch, time_steps, 2]
        return torch.stack([mem_rec_1, mem_rec_2], dim=2)

# def test_cm():
#     cm = CompetitionModel(nb_input=10, nb_ext=3, nb_inh=1, beta_ext=0.75, beta_inh=0.95, nb_decision_steps=10)
#     x = torch.rand([64, 100, 10])
#     print(cm(x).shape)
# test_cm()
            