import torch


def weight_variable(shape):
    w = torch.empty(shape)
    return torch.nn.init.xavier_uniform(w)


def bias_variable(shape):
    return torch.fill(0.1, shape=shape, dtype=torch.float32)

