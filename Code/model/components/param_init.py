import torch

def weight_variable(shape,device):
    w = torch.empty(shape)
    return torch.nn.init.xavier_uniform_(w).to(device)

def bias_variable(shape,device):
    return torch.fill(torch.empty(shape), 0.1).to(device)

