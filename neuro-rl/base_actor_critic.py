import torch
import torch.nn as nn
import numpy as np
from ratinabox.contribs.NeuralNetworkNeurons import NeuralNetworkNeurons
from fta import FTA

class BaseActorCritic(NeuralNetworkNeurons):
    """Since actors and critics have similar learning rules and trace updates we share some logic here. This is a RatInABox Neurons subclass so you can query rate maps with `.plot_rate_map()` and see history in `.history`"""

    def __init__(self, Agent, params={}):
        """Initialise the actor or critic neurons. Provide the Agent and any parameters which must include the pytorch nn.Module to use as the neural network and a list of Neurons which act as the input layers."""
        self.params = __class__.default_params.copy()
        self.params.update(params)
        super().__init__(Agent, self.params)

        self.use_eligibility_traces = self.params["eligibility_traces"]

        if self.use_eligibility_traces:
            self.initialise_traces()

        self.firingrate = self.get_state(save_torch=True) 
        self.firingrate_last = self.firingrate

        if self.params["optimizer"] is not None:
            self.optimizer = self.params["optimizer"](self.NeuralNetworkModule.parameters())
        return  
    
    def initialise_traces(self):
        """We maintain a trace of the gradients for all parameters in the network. This function initialises the traces to zero."""
        if self.use_eligibility_traces:
            self.traces = []
            for (i,param) in enumerate(self.NeuralNetworkModule.parameters()):
                shape = param.detach().numpy().shape #one trace in total 
                self.traces.append(np.zeros(shape))
        return   

    def _train_step(self, L, td_error):
        """Implements a full training step: calculates the gradients, updates the traces then steps the optimizer"""

        # self.optimizer.zero_grad()
        self._calculate_gradients(L = L)

        if self.use_eligibility_traces:
            self._update_traces(td_error=td_error)
        else:
            dt = self.Agent.dt
            for param in self.NeuralNetworkModule.parameters():
                if param.grad is not None:
                    param.grad = param.grad.detach() * td_error * dt
        
        self.optimizer.step()
        return
    
    def _calculate_gradients(self, L):
        """Calculate the gradients of L with respect to the weights. This is generic, for the critic L = V(S) (the value of the state) and for the actor L = log_prob(A | S) (the log probability of the action just taken)
    
        Args:   
            L (torch.Tensor): What to take the gradient of (must be differentiable)"""
        self.NeuralNetworkModule.zero_grad()
        L.backward(retain_graph=True)
        return

    def _update_traces(self, td_error):
        """Update the gradient traces for each output. These eligibility traces are just the gradients smoothed with an exponential kernel of timescale tau_z. We also then loop back through and """
        for (j,param) in enumerate(self.NeuralNetworkModule.parameters()):
            #trace update : z(t+dt) = (1-dt/tau_z) * z(t) + dt/tau_z * x(t) where z is the trace and x is whats being traced
            dt = self.Agent.dt; tau_z = self.tau_z
            x = param.grad.detach().numpy() #the update
            e = self.traces[j] #the trace for this output
            self.traces[j] = (1-dt/tau_z) * e + (dt/tau_z) * x
            #then set the grad to be the trace times the td error so the optimizer can access it
            param.grad = torch.tensor(self.traces[j] * td_error * self.Agent.dt, dtype=torch.float) # dt makes this update timestep invariant


# The actor and critic only different slightly in their .update() functions so we can inherit from the same base class

class Critic(BaseActorCritic):    
    def __init__(self, Agent, params={}):
        super().__init__(Agent, params) 

    def update(self, reward, train=True):
        """Accepts the reward just observed, calcuates the TD error, then updates the weights based on the gradient of its firing rate and the TD error. Finally, it updates the firing rate to reflect to new position of the Agent."""
        self._update_td_error(reward) 
        if train: super()._train_step(L = self.firingrate_torch, td_error = self.td_error)#does learning on the weights
        self.firingrate_last = self.firingrate
        super().update()  # FeedForwardLayer builtin function. 
        return

    def _update_td_error(self, reward):
        """Update the temporal difference error using the current firing rate, temporal derivative of the firing rate and the reward."""
        self.dfiringrate_dt = (self.firingrate - self.firingrate_last) / self.Agent.dt
        self.td_error = (reward + self.dfiringrate_dt - self.firingrate / self.tau).item()  # this is the continuous analog of the TD error (a scalar) 
        return

class Actor(BaseActorCritic):
    def __init__(self, Agent, params={}):
        super().__init__(Agent, params)
     # see BaseActorCritic for the default params
    def update(self, log_prob=None, td_error=None, train=True):
        """Accepts the (differentiable) log probability of the action just taken and the critic's latest TD error them updates the weights based on the gradient of the log probability and the TD error. Finally, it updates the firing rate to reflect to new position of the Agent."""
        if train: super()._train_step(L = log_prob, td_error = td_error)#does learning on the weights #does learning on the weights
        super().update()
        return
    
# CRITIC
class FTANetwork(nn.Module):
    """A FTA neural network class, default used for the core function in NeuralNetworkNeurons. 
    Specify input size, output size and hidden layer sizes (a list). Biases are used by default.

    Args:
        n_in (int, optional): The number of input neurons. Defaults to 20.
        n_out (int, optional): The number of output neurons. Defaults to 1.
        n_hidden (list, optional): A list of integers specifying the number of neurons in each hidden layer. Defaults to [20,20]."""

    def __init__(self, n_in=20, n_out=1, pre_fta=[20],post_fta=[20],input_min=-1.0, input_max=1.0, n_tiles=10,n_tilings=1, eta=1.0):
        nn.Module.__init__(self)
        fta_out = pre_fta[-1]*n_tiles*n_tilings
        n_pre = [n_in] + pre_fta 
        n_post = [fta_out] + post_fta 
        layers = nn.ModuleList()
    
        for i in range(len(n_pre)-1):
            
            layers.append(nn.Linear(n_pre[i],n_pre[i+1]))
            layers.append(nn.ReLU())
            
        layers.append(FTA(params={'n_tiles':n_tiles, 'n_tilings':n_tilings, 'fta_input_min':input_min, 'fta_input_max':input_max, 'fta_eta':eta}, input_dim=pre_fta[-1]))
        
        
        for i in range(len(n_post)-1):
            layers.append(nn.Linear(n_post[i],n_post[i+1]))
            
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        """Forward pass, X must be a torch tensor. Returns an (attached) torch tensor through which you can take gradients. """
        return self.net(X)
    
class MultiLayerPerceptron(nn.Module):
    """A generic ReLU neural network class, default used for the core function in NeuralNetworkNeurons. 
    Specify input size, output size and hidden layer sizes (a list). Biases are used by default.

    Args:
        n_in (int, optional): The number of input neurons. Defaults to 20.
        n_out (int, optional): The number of output neurons. Defaults to 1.
        n_hidden (list, optional): A list of integers specifying the number of neurons in each hidden layer. Defaults to [20,20]."""

    def __init__(self, n_in=20, n_out=1, n_hidden=[20,20]):
        nn.Module.__init__(self)
        n = [n_in] + n_hidden + [n_out]
        layers = nn.ModuleList()
        for i in range(len(n)-1):
            
            layers.append(nn.Linear(n[i],n[i+1]))
            if i < len(n)-2: layers.append(nn.ReLU()) #add a ReLU after each hidden layer (but not the last)
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        """Forward pass, X must be a torch tensor. Returns an (attached) torch tensor through which you can take gradients. """
        return self.net(X)
    
# ACTOR

class VxVyGaussian:
    """In this instance, the output of the actor is a 2 dimensional vector representing the mean of v_x and the mean of v_y (each will then be sampled from a gaussian with the same variance)."""

    def forward(self, X):
        return self.max_speed*torch.tanh(super().forward(X))

    def sample_action(self, firingrate: torch.Tensor):
        # constant std (your code overwrites previous line anyway)
        std = 0.1

        vx_dist = torch.distributions.Normal(firingrate[:, 0], std)
        vy_dist = torch.distributions.Normal(firingrate[:, 1], std)

        vx = vx_dist.sample()
        vy = vy_dist.sample()

        action = torch.stack([vx, vy], dim=-1)

        log_prob = vx_dist.log_prob(vx) + vy_dist.log_prob(vy)

        return action.detach().cpu().numpy()[0], log_prob
    
class VxVyGaussianFTA(VxVyGaussian, FTANetwork):
    
    def __init__(self,n_in, post_fta, max_speed=0.5,):
        self.n = 2
        self.max_speed = max_speed
        super().__init__(n_in = n_in,post_fta=post_fta)

class VxVyGaussianMLP(VxVyGaussian,MultiLayerPerceptron):
    """In this instance, the output of the actor is a 2 dimensional vector representing the mean of v_x and the mean of v_y (each will then be sampled from a gaussian with the same variance)."""
    def __init__(self,n_in,
                 n_hidden = [50,], 
                 max_speed=0.5,):
        self.n = 2
        self.max_speed = max_speed
        super().__init__(n_in = n_in, n_hidden=n_hidden, n_out=self.n)
