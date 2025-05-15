from __future__ import division
import os
import torch
from model.pae_gmm import PaeGmm
import numpy as np
import scipy.io as sio
import random

def set_seed(seed = 0) :
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    #print(f"Random seed set as {seed}")


if __name__ == '__main__':
    if torch.cuda.is_available() :
        print(f"CUDA is supported by this system. \nCUDA version: {torch.version.cuda}")
        dev = "cuda:0"
    else :
        dev = "cpu"

    print(f"Device : {dev}")
    device = torch.device(dev)  
    
    filename   = './Data/WebKB_texas.mat'

    #load data
    data_file = sio.loadmat(filename)
    input_data = data_file['fea']
    ground_truth = data_file['gnd']

    num_labels = len(np.unique(ground_truth))

    num_clus_r = num_labels
    num_clus_c = num_labels

    ae_config = [input_data.shape[1], 500, 200, 100, 40]

    ae_col_config = [input_data.shape[0], 500, 200, 100, 40]

    gmm_config = [[num_clus_r, 5], 40, 160, 80, 40, num_clus_r]

    epochs = 20
    epochs_pretrain = 20
    set_seed()

    machine = PaeGmm(num_clus_r, num_clus_c, ae_config, ae_col_config, gmm_config, 0, device).to(device)

    acc, nmi = machine.run(input_data, ground_truth, epochs, epochs_pretrain)

    print(f"Acc : {acc[-1]}")
    print(f"NMI : {nmi[-1]}")
