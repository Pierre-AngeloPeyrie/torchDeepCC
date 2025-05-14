import torch

def init_lin_layer(in_dim, out_dim):
    layer = torch.nn.Linear(in_dim, out_dim)
    torch.nn.init.xavier_uniform_(layer.weight)
    layer.bias.data.fill_(0.1)
    return layer
    

#def weight_variable(shape,device):
#    w = torch.empty(shape)
#    return torch.nn.init.xavier_uniform_(w).to(device)
#
#def bias_variable(shape,device):
#    return torch.fill(torch.empty(shape), 0.1).to(device)

