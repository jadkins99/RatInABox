import numpy as np
import torch
from torch.nn import Module
from base_actor_critic import FTA

def find_layer_module(net: torch.nn.Module, layer_name: str):
    '''Find layer in a network and print the type of each module as it searches'''
    layer_class = None
    
    if layer_name == 'FTA':
        layer_class = FTA
    elif layer_name == 'ReLU':
        layer_class = torch.nn.ReLU

    for m in net.modules():
        if isinstance(m, layer_class):
            return m
    raise ValueError(f"No {layer_class.__name__} module found in the network.")
    

def create_fta_hook(fta, ag, env, bins, thres=0.1):
    fta_sparsity = []
    states = []
    time_steps = []
    fta_bins_sparsity = []
    fta_bins = []
    input_arrays = []
    out_arrays = []

    
    def sparsity_function(fta_arr):
        return 1 - (np.sum(fta_arr[fta_arr > thres]) / fta_arr.size)

    def hook_fn(module, inputs, output):
        in_arr = inputs[0].detach().cpu().numpy().flatten()
        out_arr = output.detach().cpu().numpy().flatten()
        num_fta_vectors = out_arr.size // bins
        # print(f"FTA output: {out_arr}, shape: {out_arr.shape}")
        fta_b = np.array_split(out_arr, num_fta_vectors)
        # print(f"FTA input: {in_arr}, shape: {in_arr.shape}")
        # print(f"FTA bins: {fta_b}")
        # print(f"FTA output: {out_arr}, shape: {out_arr.shape}")
        # print(f"FTA bins: {fta_b} length: {len(fta_b)}")
        # fta_sparse_bins = [sparsity_function(b) for b in fta_b]
        # fta_sparsity.append(sparsity_function(out_arr))
        states.append(np.copy(ag.pos))  # or placecells.get_state()
        time_steps.append(env.t)
        # fta_bins_sparsity.append(fta_sparse_bins)
        fta_bins.append(fta_b)
        out_arrays.append(out_arr)
        input_arrays.append(in_arr)


    return hook_fn, fta_sparsity, states, time_steps,fta_bins, fta_bins_sparsity, out_arrays, input_arrays
