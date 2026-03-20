import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FTAConfiguration(object):
    """
    Configuration class remains mostly unchanged from the original,
    serving as a data structure for hyperparameters.
    """
    default_attributes = {'n_tiles': 20, 'n_tilings': 1, 'sparse_dim': None,
                          'fta_input_max': 20.0, 'fta_input_min': -20.0, 'fta_eta': 2.0,
                          'outofbound_reg': 0.0, 'extra_strength': False,
                          'individual_tiling': False,
                          'actfunctypeFTA': 'linear', 'actfunctypeFTAstrength': 'linear'}

    def __init__(self, configdict):
        for key in configdict:
            if key in self.default_attributes:
                setattr(self, key, configdict[key])

        if not hasattr(self, 'fta_input_max'):
            self.fta_input_max = self.default_attributes['fta_input_max']

        if not hasattr(self, 'fta_input_min'):
            self.fta_input_min = -self.fta_input_max
        if not hasattr(self, 'fta_eta'):
            self.fta_eta = (self.fta_input_max - self.fta_input_min) / self.n_tiles

        for key in self.default_attributes:
            if not hasattr(self, key):
                setattr(self, key, self.default_attributes[key])

class FTA(nn.Module):
    def __init__(self, params, input_dim):
        super(FTA, self).__init__()
        self.config = FTAConfiguration(params)
        
        # Configuration Setup
        self.n_tiles = int(self.config.n_tiles)
        self.n_tilings = int(self.config.n_tilings)
        self.individual_tiling = self.config.individual_tiling
        self.fta_eta = self.config.fta_eta
        self.outofbound_reg = self.config.outofbound_reg
        self.actfunctypeFTA = self.config.actfunctypeFTA
        self.extra_strength = self.config.extra_strength
        self.input_dim = input_dim

        # Define Activation Functions
        self.act_func_dict = {
            'tanh': torch.tanh,
            'linear': lambda x: x,
            'relu': F.relu,
            'sigmoid': torch.sigmoid,
            'sin': torch.sin,
            'clip': lambda x: torch.clamp(x, self.config.fta_input_min, self.config.fta_input_max)
        }

        # Extra Strength Layer
        # In TF this was set_extra_act_strength. 
        # We assume the input to this layer is the same dimension as the input to FTA.
        if self.extra_strength:
            self.strength_act = self.act_func_dict[self.config.actfunctypeFTAstrength]
            # TF used fully_connected, which implies a linear layer + activation
            self.extra_strength_layer = nn.Linear(input_dim, input_dim) 
        else:
            self.extra_strength_layer = None

        # Tiling Initialization
        if self.config.n_tilings > 1:
            c_mat, tile_delta_vector = self.get_multi_tilings(self.n_tilings, self.n_tiles)
            self.register_buffer('c_mat', torch.from_numpy(c_mat))
            self.register_buffer('tile_delta_vector', torch.from_numpy(tile_delta_vector))
        else:
            c_vec, tile_delta, low, up = self.get_tilings(
                self.n_tilings, self.n_tiles, self.config.fta_input_min, self.config.fta_input_max)
            # Register buffers so they are saved with model and moved to device
            self.register_buffer('c_vec', torch.from_numpy(c_vec))
            self.register_buffer('tile_delta', torch.tensor(tile_delta))
            self.register_buffer('tiling_low_bound', torch.tensor(low))
            self.register_buffer('tiling_up_bound', torch.tensor(up))

        print(f' fta_eta: {self.fta_eta}, n_tilings: {self.n_tilings}, n_tiles: {self.n_tiles}, input_dim: {self.input_dim}')

    def get_tilings(self, n_tilings, n_tile, input_min, input_max):
        tile_delta = (input_max - input_min) / n_tile
        if n_tilings == 1:
            one_c = np.linspace(input_min, input_max, n_tile, endpoint=False).astype(np.float32)
            return one_c, tile_delta, input_min, input_max
        
        # Logic for multiple tilings (used if individual_tiling=False but n_tilings=1 is handled above)
        maxoffset = n_tilings * (input_max - input_min) / n_tile
        tiling_length = input_max - input_min + maxoffset
        startc = input_min - np.random.uniform(0, maxoffset, n_tilings)
        
        # This part of the logic handles generating the list, 
        # but the actual buffer registration happens in __init__
        # Simplified here to return what is needed for single tiling as that is the standard path

        # added following code because the original code had it
        # c_list = []
        # for n in range(n_tilings):
        #     step = tiling_length / n_tile
        #     one_c = torch.arange(startc[n], startc[n] + tiling_length, step, dtype=torch.float32)
        #     c_list.append(one_c)
        # tiling_low_bound = np.min(startc) - maxoffset
        # tiling_up_bound = np.max(startc) + tiling_length
        # return c_list, tile_delta, tiling_low_bound, tiling_up_bound
        return None
        
    # def get_multi_tilings(self, n_tilings, n_tile):
    #     input_max_list = np.random.choice(self.config.fta_input_max, n_tilings)

    #     c_list = []
    #     tile_delta_list = []

    #     for n in range(n_tilings):
    #         ind = n % len(input_max_list)

    #         one_c = np.linspace(
    #             -input_max_list[ind],
    #             input_max_list[ind],
    #             n_tile,
    #             endpoint=False
    #         ).astype(np.float32)

    #         # Equivalent to tf.constant(...).reshape((-1, n_tile))
    #         one_c_tensor = torch.tensor(one_c.copy(), dtype=torch.float32).reshape(-1, n_tile)

    #         c_list.append(one_c_tensor)

    #         tile_delta_list.append(one_c[1] - one_c[0])

    #     # Equivalent to tf.concat(axis=0)
    #     c_mat = torch.cat(c_list, dim=0)

    #     # Equivalent to tf.reshape(..., [n_tilings, 1])
    #     tile_delta_vector = torch.tensor(
    #         np.array(tile_delta_list).astype(np.float32),
    #         dtype=torch.float32
    #     ).reshape(n_tilings, 1)

    #     return c_mat, tile_delta_vector

    def get_multi_tilings(self, n_tilings, n_tile):
        # Handle fta_input_max list logic
        input_max_val = self.config.fta_input_max
        if not isinstance(input_max_val, list):
             input_max_val = [input_max_val]
             
        input_max_list = np.random.choice(input_max_val, n_tilings)
        c_list = []
        tile_delta_list = []
        
        for n in range(n_tilings):
            ind = n % len(input_max_list)
            limit = input_max_list[ind]
            one_c = np.linspace(-limit, limit, n_tile, endpoint=False).astype(np.float32)
            c_list.append(one_c.reshape((-1, n_tile)))
            tile_delta_list.append(one_c[1] - one_c[0])
            
        c_mat = np.concatenate(c_list, axis=0).astype(np.float32)
        tile_delta_vector = np.array(tile_delta_list).astype(np.float32).reshape(n_tilings, 1)
        return c_mat, tile_delta_vector

    def Iplus_eta(self, x, eta):
        # TF: tf.cast(x <= eta) * x + tf.cast(x > eta)
        # If eta is 0, return sign(x)
        if isinstance(eta, (int, float)) and eta == 0:
            return torch.sign(x)
            
        # Logic: if x <= eta, return x. If x > eta, return 1.0.
        # Note: In TF code, the second term was `tf.cast(x > eta)` which results in 1.0 or 0.0.
        # It did not multiply by x.
        condition = (x <= eta)
        return condition.float() * x + (~condition).float()

    def _sum_relu(self, c, x, delta):
        return F.relu(c - x) + F.relu(x - delta - c)

    def _get_strength(self, raw_input):
        if not self.extra_strength:
            return 1.0
        # raw_input is the input to the layer before FTA
        out = self.extra_strength_layer(raw_input)
        out = self.strength_act(out)
        return out.unsqueeze(-1) # Add dimension for broadcasting

    def compute_out_of_bound_loss(self, x):
        loss = 0.0
        if self.outofbound_reg > 0 and self.actfunctypeFTA not in ['tanh', 'sigmoid', 'clip']:
            # Pytorch requires explicit loss handling. We usually return this or add to a global tracker.
            # Here we return it as a secondary output of forward, or store it.
            upper_violation = (x > self.tiling_up_bound).float() * x
            lower_violation = (x < self.tiling_low_bound).float() * x
            loss = torch.mean(torch.sum(upper_violation, dim=1)) - \
                   torch.mean(torch.sum(lower_violation, dim=1))
            loss = self.outofbound_reg * loss
        return loss

    def forward(self, raw_input):
        """
        raw_input: tensor of shape (batch_size, input_dim)
        """
        # Pre-FTA Activation
        x = self.act_func_dict[self.actfunctypeFTA](raw_input)
        
        # Track loss
        out_of_bound_loss = self.compute_out_of_bound_loss(x)
        self.last_loss = out_of_bound_loss # Store for access later

        d = x.shape[1]
        
        # Route to specific implementation
        if self.n_tilings > 1:
            if self.individual_tiling:
                return self._forward_individual(x, raw_input, d)
            else:
                return self._forward_multi(x, raw_input, d)
        else:
            return self._forward_single(x, raw_input, d)

    def _forward_single(self, x, raw_input, d):
        # x shape: [Batch, d] -> [Batch, d, 1]
        x_reshaped = x.view(-1, d, 1)
        
        # Calculate membership
        # c_vec shape broadcast
        val = self._sum_relu(self.c_vec, x_reshaped, self.tile_delta)
        val = self.Iplus_eta(val, self.fta_eta)
        
        strength = self._get_strength(raw_input)
        
        onehot = (1.0 - val) * strength
        # Reshape to [Batch, d * n_tiles]
        return onehot.view(-1, d * self.n_tiles)

    def _forward_multi(self, x, raw_input, d):
        # x shape: [Batch, d] -> [Batch, d, 1, 1]
        x_reshaped = x.view(-1, d, 1, 1)
        
        val = self._sum_relu(self.c_mat, x_reshaped, self.tile_delta_vector)
        val = self.Iplus_eta(val, self.tile_delta_vector)
        
        onehots = 1.0 - val
        return onehots.view(-1, d * self.n_tiles * self.n_tilings)

    def _forward_individual(self, x, raw_input, d):
        # x shape: [Batch, d] -> [Batch, d, 1]
        x_reshaped = x.view(-1, d, 1)
        
        val = self._sum_relu(self.c_mat, x_reshaped, self.tile_delta_vector)
        val = self.Iplus_eta(val, self.tile_delta_vector)
        
        onehots = 1.0 - val
        return onehots.view(-1, d * self.n_tiles)

class FTA_QNN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden1, n_hidden2, fta_params):
        super(FTA_QNN, self).__init__()
        
        # Layer 1
        self.fc1 = nn.Linear(n_input, n_hidden1)
        
        # Layer 2 (The Sparse Layer)
        # Note: In the TF code, hidden1 goes into "sparse_phi".
        # sparse_phi = dense(hidden1, n_hidden2) -> FTA
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        
        # Initialize FTA module with configuration
        # Pass n_hidden2 as the input dimension to the FTA logic
        self.fta_layer = FTA(fta_params, input_dim=n_hidden2)
        
        # Output Layer
        # The input to this layer is the output of FTA.
        # We need to calculate the size.
        # FTA output size = n_hidden2 * n_tiles * n_tilings
        fta_out_dim = n_hidden2 * self.fta_layer.n_tiles * self.fta_layer.n_tilings
        
        self.fc3 = nn.Linear(fta_out_dim, n_output)
        
        # Initialize Output Weights (TF used random uniform -0.003 to 0.003)
        nn.init.uniform_(self.fc3.weight, -0.003, 0.003)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, state_input):
        # Layer 1
        hidden1 = F.relu(self.fc1(state_input))
        
        # Layer 2 (Pre-activation)
        phi_pre = self.fc2(hidden1)
        
        # Apply FTA
        # We pass phi_pre. If extra_strength is on, FTA might need hidden1 (the previous input)
        # However, standard FTA usually applies strength based on its own input.
        # Based on TF code: SparseActFunc.set_extra_act_strength(hidden1...)
        # This implies strength is derived from hidden1.
        # To support this strict logic, we would need to pass hidden1 into FTA.
        # For this implementation, I updated FTA._get_strength to take `raw_input`.
        # We will pass `phi_pre` as the primary input. 
        # If strict adherence to TF 'extra_strength' on 'hidden1' is needed, 
        # we would pass hidden1 as a secondary argument.
        
        sparse_phi = self.fta_layer(phi_pre)
        
        # Output Layer
        q_values = self.fc3(sparse_phi)
        
        # Ops
        max_qvalue, max_ind = torch.max(q_values, dim=1)
        
        return q_values, max_qvalue, max_ind, sparse_phi




# ------------------ Usage Example --------------------

def usage_example():
    # Configuration
    fta_config = {
        'n_tiles': 20, 
        'n_tilings': 1, 
        'fta_input_max': 20.0,
        'extra_strength': False
    }

    n_input = 4
    n_output = 2
    n_hidden1 = 64
    n_hidden2 = 32

    # Instantiate Model
    model = FTA_QNN(n_input, n_output, n_hidden1, n_hidden2, fta_config)
    print("Model Architecture:")
    print(model)

    # Dummy Input
    dummy_input = torch.randn(5, n_input) # Batch size 5

    # Forward Pass
    q_vals, max_q, max_idx, sparse_rep = model(dummy_input)

    print("\nForward Pass Results:")
    print(f"Q-Values Shape: {q_vals.shape}")
    print(f"Sparse Representation Shape: {sparse_rep.shape}")
    
    # Check for OOB loss if needed
    if hasattr(model.fta_layer, 'last_loss'):
         print(f"FTA OOB Loss: {model.fta_layer.last_loss}")

if __name__ == "__main__":
    usage_example()