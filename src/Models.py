import torch, sqlite3
import snntorch as snn
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.nn.utils import parameters_to_vector
from dataclasses import dataclass

@dataclass
class RandmanSNNConfig:
    nb_hidden_1: int
    nb_hidden_2: int
    beta: float
    
    # first hidden layer
    learn_beta_1: bool = None
    learn_beta_per_neuron_1: bool = None
    recurrent: bool = None
    learn_weights_1: bool = None
    
    # output layer
    learn_beta_out: bool = None
    learn_beta_per_neuron_out: bool = None
    learn_weights_out: bool = None
    
    # low-rank approximation
    synapse_rank: int = -1  # -1 means no low-rank, full weight matrix
    
    @classmethod
    def lookup_by_id(cls, id, db_path='data/landscape-analysis.db'):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()             
        cursor.execute(f"SELECT nb_hidden_1, nb_hidden_2, beta, learn_beta_1, learn_beta_per_neuron_1, recurrent, learn_weights_1, learn_beta_out, learn_beta_per_neuron_out, learn_weights_out, synapse_rank FROM snn_models WHERE id={id}")
        row = list(cursor.fetchone())
        conn.close()
        
        if row is None:
            raise ValueError(f"No RandmanSNN configuration found with id {id}")
        
        # convert int to bool
        row[3] = bool(row[3]) # learn_beta
        row[4] = bool(row[4]) # learn_beta_per_neuron
        row[5] = bool(row[5]) # recurrent
        row[6] = bool(row[6]) # learn_weights
        row[7] = bool(row[7]) # learn_beta_out
        row[8] = bool(row[8]) # learn_beta_per_neuron_out
        row[9] = bool(row[9]) # learn_weights_out

        return cls(*row)
    
    @property
    def hidden_1_train_scheme(self):
        """
        Train_scheme is determined by learn_beta_1, learn_beta_per_neuron_1, and learn_weights_1. 

        Return:
            'weights': only weights are learned
            'beta-per-neuron': only beta per neuron is learned
            'weights+beta-per-neuron': both weights and beta per neuron are learned
        """
        scheme = None
        if not self.learn_weights_1 and not self.learn_beta_1:
            scheme = 'none'
        elif self.learn_weights_1 and not self.learn_beta_1:
            scheme = 'weights'
        elif not self.learn_weights_1 and self.learn_beta_1 and self.learn_beta_per_neuron_1:
            scheme = 'beta-per-neuron'
        elif self.learn_weights_1 and self.learn_beta_1 and self.learn_beta_per_neuron_1:
            scheme = 'weights+beta-per-neuron'
        else:
            raise ValueError(f"Invalid configuration: learn_weights={self.learn_weights_1}, learn_beta={self.learn_beta_1}, learn_beta_per_neuron={self.learn_beta_per_neuron_1}. This combination is not supported.")
        return scheme
    
    @hidden_1_train_scheme.setter
    def hidden_1_train_scheme(self, scheme):
        if scheme == 'none':
            self.learn_weights_1 = False
            self.learn_beta_1 = False
            self.learn_beta_per_neuron_1 = False
        elif scheme == 'weights':
            self.learn_weights_1 = True
            self.learn_beta_1 = False
            self.learn_beta_per_neuron_1 = False
        elif scheme == 'beta-per-neuron':
            self.learn_weights_1 = False
            self.learn_beta_1 = True
            self.learn_beta_per_neuron_1 = True
        elif scheme == 'weights+beta-per-neuron':
            self.learn_weights_1 = True
            self.learn_beta_1 = True
            self.learn_beta_per_neuron_1 = True
        else:
            raise ValueError(f"Invalid train_scheme: {scheme}. Supported schemes are 'weights', 'beta-per-neuron', and 'weights+beta-per-neuron'.")
        

    @property
    def out_train_scheme(self):
        """
        Train_scheme is determined by learn_beta_out, learn_beta_per_neuron_out, and learn_weights_out. 

        Return:
            'weights': only weights are learned
            'beta-per-neuron': only beta per neuron is learned
            'weights+beta-per-neuron': both weights and beta per neuron are learned
        """
        scheme = None
        if not self.learn_weights_out and not self.learn_beta_out:
            scheme = 'none'
        elif self.learn_weights_out and not self.learn_beta_out:
            scheme = 'weights'
        elif not self.learn_weights_out and self.learn_beta_out and self.learn_beta_per_neuron_out:
            scheme = 'beta-per-neuron'
        elif self.learn_weights_out and self.learn_beta_out and self.learn_beta_per_neuron_out:
            scheme = 'weights+beta-per-neuron'
        else:
            raise ValueError(f"Invalid configuration: learn_weights_out={self.learn_weights_out}, learn_beta_out={self.learn_beta_out}, learn_beta_per_neuron_out={self.learn_beta_per_neuron_out}. This combination is not supported.")
        return scheme

    @out_train_scheme.setter
    def out_train_scheme(self, scheme):
        if scheme == 'none':
            self.learn_weights_out = False
            self.learn_beta_out = False
            self.learn_beta_per_neuron_out = False
        elif scheme == 'weights':
            self.learn_weights_out = True
            self.learn_beta_out = False
            self.learn_beta_per_neuron_out = False
        elif scheme == 'beta-per-neuron':
            self.learn_weights_out = False
            self.learn_beta_out = True
            self.learn_beta_per_neuron_out = True
        elif scheme == 'weights+beta-per-neuron':
            self.learn_weights_out = True
            self.learn_beta_out = True
            self.learn_beta_per_neuron_out = True
        else:
            raise ValueError(f"Invalid train_scheme: {scheme}. Supported schemes are 'weights', 'beta-per-neuron', and 'weights+beta-per-neuron'.")

    def write_to_db(self, db_path='data/landscape-analysis.db'):
        if None in (self.learn_beta_1, self.learn_beta_per_neuron_1, self.recurrent, self.learn_weights_1, self.learn_beta_out, self.learn_weights_out, self.learn_beta_per_neuron_out, self.synapse_rank):
            raise ValueError("All boolean fields (learn_beta, learn_beta_per_neuron, recurrent, learn_weights, learn_beta_out, learn_weights_out, learn_beta_per_neuron_out, synapse_rank) must be set before writing to the database.")

        if self.nb_hidden_2 is None or (self.nb_hidden_2 < 1 and self.nb_hidden_2 != -1):
            self.nb_hidden_2 = -1
            print(f"Warning: nb_hidden_2 is set to {self.nb_hidden_2}. This is not a valid value for SNN models. It will be set to -1.")
            
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        # Insert the new configuration into the database
        try:
            cur.execute(
                """INSERT INTO snn_models (nb_hidden_1, nb_hidden_2, beta, learn_beta_1, learn_beta_per_neuron_1, recurrent, learn_weights_1, learn_beta_out, learn_weights_out, learn_beta_per_neuron_out, synapse_rank)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.nb_hidden_1,
                    self.nb_hidden_2,
                    self.beta,
                    int(self.learn_beta_1),
                    int(self.learn_beta_per_neuron_1),
                    int(self.recurrent),
                    int(self.learn_weights_1),
                    int(self.learn_beta_out),
                    int(self.learn_weights_out),
                    int(self.learn_beta_per_neuron_out),
                    self.synapse_rank
                )
            )
        except sqlite3.IntegrityError:
            print(f"Configuration {self} already exists in the database.")
        con.commit()
        con.close()
    
    def get_id(self, db_path='data/landscape-analysis.db'):
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        # Get the id of the configuration
        cur.execute(
            """SELECT id FROM snn_models WHERE nb_hidden_1=? AND nb_hidden_2=? AND beta=? AND learn_beta_1=? AND learn_beta_per_neuron_1=? AND recurrent=? AND learn_weights_1=? AND learn_beta_out=? AND learn_weights_out=? AND learn_beta_per_neuron_out=? AND synapse_rank=?""",
            (self.nb_hidden_1, self.nb_hidden_2, self.beta, int(self.learn_beta_1), int(self.learn_beta_per_neuron_1), int(self.recurrent), int(self.learn_weights_1), int(self.learn_beta_out), int(self.learn_weights_out), int(self.learn_beta_per_neuron_out), self.synapse_rank)
        )
        
        row = cur.fetchone()
        con.close()
        
        if row is None:
            raise ValueError(f"Configuration {self} not found in the database. Notice that -1 is used to indicate no 2nd hidden layer.")

        return row[0]

import torch
import snntorch as snn
import torch.nn as nn
from torch.nn.functional import cross_entropy
from snntorch import surrogate
from torch.nn.utils import parameters_to_vector

device = 'cuda'

class FrozenLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # init records
        self.in_features = in_features
        self.out_features = out_features
        
        # init weights
        w = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(w)
        self.register_buffer("weight", w)

        if bias:
            bound = 1 / in_features**0.5
            b = torch.empty(out_features).uniform_(-bound, bound)

            self.register_buffer("bias", b)

    def forward(self, x):
        out = x @ self.weight.T
        if hasattr(self, "bias"):
            out += self.bias
        return out

class RandmanSNN(nn.Module):
    def _setup_fc1(self, num_inputs, num_hidden_1, learn_weights, synapse_rank):
        FCType = nn.Linear if learn_weights else FrozenLinear
        if synapse_rank == -1 or synapse_rank >= min(num_inputs, num_hidden_1):
            self.fc1 = FCType(num_inputs, num_hidden_1, bias=False)
        else:
            self.fc1_u = FCType(num_inputs, synapse_rank, bias=False)
            self.fc1_v = FCType(synapse_rank, num_hidden_1, bias=False)

    def _setup_fc_out(self, num_hidden_1, num_outputs, learn_weights, nb_hidden_2):
        FCType = nn.Linear if learn_weights else FrozenLinear
        input_size = num_hidden_1 if nb_hidden_2 == -1 else nb_hidden_2
        self.fc_out = FCType(input_size, num_outputs, bias=False)

    def __init__(self, num_inputs, num_outputs, snn_config: RandmanSNNConfig, spike_grad=None):
        super(RandmanSNN, self).__init__()
        self.recurrent = snn_config.recurrent
        self.nb_hidden_1 = snn_config.nb_hidden_1  
        # If no spike_grad specified, use default
        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid(slope=25)

        # 1st hidden layer
        self._setup_fc1(num_inputs, snn_config.nb_hidden_1, snn_config.learn_weights_1, snn_config.synapse_rank)
        # add recurrent layer if needed
        if self.recurrent:
            self.fc1_rec = nn.Linear(snn_config.nb_hidden_1, snn_config.nb_hidden_1, bias=False) if snn_config.learn_weights_1 else FrozenLinear(snn_config.nb_hidden_1, snn_config.nb_hidden_1, bias=False)
        # add LIF layer
        self.lif1 = snn.Leaky(
            beta=snn_config.beta * torch.ones(snn_config.nb_hidden_1) if snn_config.learn_beta_per_neuron_1 else snn_config.beta,
            learn_beta=snn_config.learn_beta_1,
            spike_grad=spike_grad
        )

        # 2nd hidden layer (optional)
        if snn_config.nb_hidden_2 < 1 and snn_config.nb_hidden_2 != -1:
            raise ValueError(f"Invalid value for nb_hidden_2: {snn_config.nb_hidden_2}. It should be -1 or a positive integer.")
        # 2nd hidden layer
        if snn_config.nb_hidden_2 != -1:
            self.fc2 = nn.Linear(snn_config.nb_hidden_1, snn_config.nb_hidden_2, bias=False) if snn_config.learn_weights_1 else FrozenLinear(snn_config.nb_hidden_1, snn_config.nb_hidden_2, bias=False)
            # recurrent layer
            if self.recurrent:
                self.fc2_rec = nn.Linear(snn_config.nb_hidden_2, snn_config.nb_hidden_2, bias=False) if snn_config.learn_weights_1 else FrozenLinear(snn_config.nb_hidden_2, snn_config.nb_hidden_2, bias=False)
            # LIF layer
            self.lif2 = snn.Leaky(
            beta=snn_config.beta * torch.ones(snn_config.nb_hidden_2) if snn_config.learn_beta_per_neuron_1 else snn_config.beta,
            learn_beta=snn_config.learn_beta_1,
            spike_grad=spike_grad
            )

        # Output layer
        self._setup_fc_out(snn_config.nb_hidden_1, num_outputs, snn_config.learn_weights_out, snn_config.nb_hidden_2)
        self.lif_out = snn.Leaky(beta=snn_config.beta * torch.ones(num_outputs) if snn_config.learn_beta_per_neuron_out else snn_config.beta, learn_beta=snn_config.learn_beta_out, spike_grad=spike_grad, reset_mechanism='none')

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.zeros_(m.bias)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        batch_size, time_steps, num_neurons = x.shape
        x = x.permute(1, 0, 2)  # (time, batch, neurons)

        ## Initialize membrane potentials and spikes
        self.mem1_rec = [torch.zeros(batch_size, self.nb_hidden_1, device=x.device),]
        self.spk1_rec = [torch.zeros(batch_size, self.nb_hidden_1, device=x.device),]
        if hasattr(self, 'fc2'):
            self.mem2_rec = [torch.zeros(batch_size, self.fc2.out_features, device=x.device),]
            self.spk2_rec = [torch.zeros(batch_size, self.fc2.weight.shape[0], device=x.device),]
        self.mem_out_rec = [torch.zeros(batch_size, self.fc_out.out_features, device=x.device),]

        for t in range(time_steps):
            # First hidden layer, either with or without low-rank
            if hasattr(self, 'fc1'):
                input_1 = self.fc1(x[t])
            else: 
                input_1 = self.fc1_v(self.fc1_u(x[t]))
            if self.recurrent:
                input_1 += self.fc1_rec(self.spk1_rec[-1])
            spk1, mem1 = self.lif1(input_1, self.mem1_rec[-1])
            self.spk1_rec.append(spk1)
            self.mem1_rec.append(mem1)
            
            # Second hidden layer if it exists
            if hasattr(self, 'fc2'):
                input_2 = self.fc2(spk1)
                if self.recurrent:
                    input_2 += self.fc2_rec(self.spk2_rec[-1])
                spk2, mem2 = self.lif2(input_2, self.mem2_rec[-1])
                self.spk2_rec.append(spk2)
                self.mem2_rec.append(mem2)

            # Output layer
            _, mem_out = self.lif_out(self.fc_out(spk2 if hasattr(self, 'fc2') else spk1), self.mem_out_rec[-1])          
            self.mem_out_rec.append(mem_out)

        # Each stacked item has shape [time_steps, batch_size, num_neurons]
        # Including the first tensor kills the learning in SGD, very interesting
        return torch.stack(self.spk1_rec[1:], dim=0), torch.stack(self.mem_out_rec[1:], dim=0)

def calc_randmansnn_parameters(nb_inputs, nb_outputs, snn_config: RandmanSNNConfig):
    model = RandmanSNN(num_inputs=nb_inputs, num_outputs=nb_outputs, snn_config=snn_config).to('cpu')
    vector = parameters_to_vector(model.parameters())
    return vector.size(0)

def spike_regularized_cross_entropy(logits, y, spikes, lambda_reg=1e-3):
    loss_ce = cross_entropy(logits, y)
    rate_loss = lambda_reg * spikes.sum(dim=0).mean()
    if torch.isnan(loss_ce) or torch.isnan(rate_loss):
        print("NaN detected in loss computation:", loss_ce.item(), rate_loss.item())
    return loss_ce + rate_loss

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
        self.mem = self.lif.init_leaky().to(self.input_fc.weight.device)
        self.last_spks_queue = [torch.zeros(self.get_nb_neurons(), device=self.input_fc.weight.device) for _ in range(nb_decision_steps if nb_decision_steps != None else len(self.last_spks_queue))]  
        
        # readout neuron
        self.readout_mem = self.readout_lif.init_leaky().to(self.input_fc.weight.device)
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
        x = torch.cat([x, torch.zeros(5 + self.nb_decision_steps, x.shape[1], x.shape[2], device=x.device)])
        
        # initalize membrane potentials
        self.init_states()
        
        # init spikes with shape [nb_neurons]. The batch size will be broadcasted
        inh_spk = torch.zeros([self.get_nb_inh()], device=x.device)
        
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
            