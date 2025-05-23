import model.components.stat_lib as slib
import model.components.param_init as pini

import torch
import torch.nn as nn
import torch.nn.functional as F



class GMMEstimationNetRaw(nn.Module):
    def __init__(self, config, device):
        super(GMMEstimationNetRaw,self).__init__()
        # DMM config
        self.dmm_config = config
        self.num_mixture = self.dmm_config[0][0]
        self.gmm_layer = self.dmm_config[0][1]
        # Layer 1
        self.wi = []
        self.bi = []
        for i in range(1, len(self.dmm_config) - 1):
            w = pini.weight_variable([self.dmm_config[i], self.dmm_config[i+1]], device)
            b = pini.bias_variable([self.dmm_config[i+1]], device)
            self.wi.append(w)
            self.bi.append(b)
        # Mixture modeling
        self.gmm_config = [self.num_mixture, self.dmm_config[self.gmm_layer], self.dmm_config[self.gmm_layer]]
        self.gmm = slib.GaussianMixtureModeling(self.gmm_config)

    def run(self, x, keep_prob):
        # Mixture estimation network
        z = [x]
        for i in range(0, len(self.wi)):
            if i == len(self.wi) - 1:
                zi = F.dropout(z[i], p=1 - (keep_prob))
            else:
                zi = z[i]
            if i < len(self.wi) - 1:
                zj = F.tanh(torch.matmul(zi, self.wi[i]) + self.bi[i]) # tanh, softsign, sigmoid, softplus
            else:
                zj = F.softmax(torch.matmul(zi, self.wi[i]) + self.bi[i],dim=1)
                # f = torch.matmul(zi, self.wi[i]) + self.bi[i]
            z.append(zj)
        

        p = z[len(z)-1]
        f = z[self.gmm_layer-1] # the representation after 'softmax'
        #f = x # the output of AE

        #print'f'
        #print(f)

        # Log likelihood
        #gmm_energy, pen_dev, likelihood, _, _, _, _ = self.gmm.eval(f, p)
        gmm_energy, pstr, pen_dev, likelihood, phi, x_t, p_t, z_p, z_t, mixture_mean, mixture_dev, mixture_cov, mixture_dev_det = self.gmm.vi_learning(f, p)
        # k_dist, pen_dev, mixture_dev, t1 = self.kmm.eval(x, p)
        # mixture_dev_0 = mixture_dev[:, 0]
        # prior_energy, prior_energy_sum = self.inverse_gamma.eval(mixture_dev, phi)
        # train
        # energy = gmm_energy + prior_energy_sum
        loss = gmm_energy
        # loss = k_dist
        
        #return loss, pen_dev, likelihood, p
        return loss, pen_dev, likelihood, pstr, x_t, p_t, z_p, z_t, mixture_mean, mixture_dev, mixture_cov, mixture_dev_det


    def model(self, x):
        # Mixture estimation network
        z = [x]
        for i in range(0, len(self.wi)):
            zi = z[i]
            if i < len(self.wi) - 1:
                zj = F.tanh(torch.matmul(zi, self.wi[i]) + self.bi[i])
            else:
                zj = F.softmax(torch.matmul(zi, self.wi[i]) + self.bi[i],dim=1)
            z.append(zj)
        p = z[len(z) - 1]
        f = z[self.gmm_layer - 1]
        # Log likelihood
        _, _, _, phi, mixture_mean, mixture_dev, mixture_cov = self.gmm.eval(f, p)
        return phi, mixture_mean, mixture_dev, mixture_cov

    def test(self, x, phi, mixture_mean, mixture_dev, mixture_cov):
        # Mixture estimation network
        z = [x]
        for i in range(0, len(self.wi)):
            zi = z[i]
            if i < len(self.wi) - 1:
                zj = F.tanh(torch.matmul(zi, self.wi[i]) + self.bi[i])
            else:
                zj = F.softmax(torch.matmul(zi, self.wi[i]) + self.bi[i],dim=1)
            z.append(zj)
        f = z[self.gmm_layer - 1]
        likelihood = self.gmm.test(f, phi, mixture_mean, mixture_dev, mixture_cov)
        return likelihood, f
