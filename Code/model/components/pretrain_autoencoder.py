import torch
import torch.nn.functional as F

from model.components.param_init import init_lin_layer


class PretrainAutoencoder(torch.nn.Module):
    def __init__(self, config, num_drop_out):
        super(PretrainAutoencoder,self).__init__()
        self.num_dim = config
        self.code_layer = len(config)
        self.num_dropout_layer = num_drop_out
        # Parameters in layers
        self.layers = torch.nn.ModuleList([])
        # Encode
        for i in range(0, len(self.num_dim)-1):
            self.layers.append(init_lin_layer(self.num_dim[i], self.num_dim[i+1]))
        # Decode
        for i in range(1, len(self.num_dim)):
            j = len(self.num_dim)-i
            self.layers.append(init_lin_layer(self.num_dim[j], self.num_dim[j-1]))

        self.optimizer = torch.optim.Adam(self.parameters(),1e-4)
        
    def forward(self, x, keep_prob):
        vision_coef = 1.0
        error = []
        reg = []
        l2_reg = torch.tensor(0)

        # Encode
        zi = x
        for i in range(int(len(self.layers)/2)):
            if i < len(self.layers) / 2 - 1:
                zj = F.tanh(self.layers[i](zi))
            else:
                zj = self.layers[i](zi)
            if i < self.num_dropout_layer:
                zj = F.dropout(zj, p = 1 - (keep_prob))
            ni = len(self.layers) - i - 1
            if i == 0:
                z_r = self.layers[len(self.layers) - 1](zj)
            else:
                z_r = F.tanh(self.layers[ni](zj))
            error_l = torch.mean(torch.linalg.norm(zi - z_r, ord = 2, dim=1, keepdims=True))
            error.append(error_l * vision_coef)
            zi = zj
            reg.append(self.layers[i].weight.square().sum()/2 + self.layers[ni].weight.square().sum()/2)
            l2_reg = l2_reg + self.layers[i].weight.square().sum()/2
        zc = zi
        
        # Decode
        for i in range(int(len(self.layers)/2), len(self.layers)):
            if i < len(self.layers) - 1:
                zj = F.tanh(self.layers[i](zi))
                if i >= len(self.layers) - 1 - self.num_dropout_layer:
                    zj = F.dropout(zj, p = 1 - (keep_prob))
            else:
                zj = self.layers[i](zi)
            l2_reg = l2_reg + self.layers[i].weight.square().sum()/2
            zi = zj
        zo = zi

        # Cosine similarity
        loss = torch.linalg.norm(x - zo, ord=2, dim=1, keepdims=True)
        
        xo = torch.concat([zc], 1)

        error.append(torch.mean(loss))

        return xo, error, l2_reg, reg
    
    def fit(self, x, epochs, keep_prob):
        for i in range(epochs) :
            self.train()
            train_z, train_error, train_l2_reg, train_reg = self(x, keep_prob)
            self.optimizer.zero_grad()
            for j in range(len(train_error) - 1):
                loss = train_error[j] * 5e0 + train_reg[j] * 1e0
                loss.backward(retain_graph = True)
            self.optimizer.step()
            if i % 20 == 0 : print(f'loss pretrain row AE round {i} : {train_error[-1]}')
        return train_z, train_error, train_l2_reg

    def test(self, x):
        # Encode
        zi = x
        for i in range(int(len(self.wi) / 2)):
            if i < len(self.wi) / 2 - 1:
                zj = F.tanh(torch.matmul(zi, self.wi[i]) + self.bi[i])
            else:
                zj = torch.matmul(zi, self.wi[i]) + self.bi[i]
            zi = zj
        zc = zi
        # Decode
        for i in range(int(len(self.wi) / 20), len(self.wi)):
            if i < len(self.wi) - 1:
                zj = F.tanh(torch.matmul(zi, self.wi[i]) + self.bi[i])
            else:
                zj = torch.matmul(zi, self.wi[i]) + self.bi[i]
            zi = zj
        zo = zi

        # Cosine similarity
        normalize_x = F.normalize(x, dim=1)
        normalize_zo = F.normalize(zo, dim=1)
        cos_sim = torch.sum(torch.multiply(normalize_x, normalize_zo), 1, keepdims=True)
        dist = torch.linalg.norm(x - zo, ord=2, dim=1, keepdims=True)
        # relative_dist = dist / torch.linalg.norm(x, ord=2, dim=1, keep_dims=True)
        # dist_norm = (dist - self.dist_min) / (self.dist_max - self.dist_min + 1e-12)
        # xo = torch.concat([zc, relative_dist, cos_sim], 1)
        # xo = torch.concat([zc, relative_dist], 1)
        return dist

    def dcn_run(self, x, keep_prob):
        vision_coef = 1.0
        error = []
        var_list = []
        reg = []
        l2_reg = 0
        # Encode
        zi = x
        for i in range(int(len(self.wi) / 2)):
            if i < len(self.wi) / 2 - 1:
                zj = F.tanh(torch.matmul(zi, self.wi[i]) + self.bi[i])
            else:
                zj = torch.matmul(zi, self.wi[i]) + self.bi[i]
            if i < self.num_dropout_layer:
                zj = F.dropout(zj, p = 1 - (keep_prob))
            ni = len(self.wi) - i - 1
            if i == 0:
                z_r = torch.matmul(zj, self.wi[len(self.wi) - 1]) + self.bi[len(self.bi) - 1]
            else:
                z_r = F.tanh(torch.matmul(zj, self.wi[ni]) + self.bi[ni])
            error_l = torch.mean(torch.linalg.norm(zi - z_r, ord=2, dim=1, keepdims=True))
            error.append(error_l * vision_coef)
            zi = zj
            reg.append(self.wi[i].square().sum()/2 + self.wi[ni].square().sum()/2)
            l2_reg = l2_reg + self.wi[i].square().sum()/2
            var_list.append([self.wi[i], self.bi[i], self.wi[ni], self.bi[ni]])
        zc = zi
        # Decode
        for i in range(int(len(self.wi) / 2), len(self.wi)):
            if i < len(self.wi) - 1:
                zj = F.tanh(torch.matmul(zi, self.wi[i]) + self.bi[i])
                if i >= len(self.wi) - 1 - self.num_dropout_layer:
                    zj = F.dropout(zj, p=1 - (keep_prob))
            else:
                zj = torch.matmul(zi, self.wi[i]) + self.bi[i]
            l2_reg = l2_reg + self.wi[i].square().sum()/2
            zi = zj
        zo = zi

        # Cosine similarity
        normalize_x = F.normalize(x, dim=1)
        normalize_zo = F.normalize(zo, dim=1)
        cos_sim = torch.sum(torch.multiply(normalize_x, normalize_zo), dim=1, keepdims=True)
        loss = torch.linalg.norm(x - zo, ord=2, dim=1, keepdims=True)
        dist = torch.linalg.norm(x - zo, ord=2, dim=1, keepdims=True)
        relative_dist = dist / torch.linalg.norm(x, ord=2, dim=1, keepdims=True)
        # self.dist_min = torch.min(dist)
        # self.dist_max = torch.max(dist)
        # dist_norm = (dist - self.dist_min) / (self.dist_max - self.dist_min + 1e-12)
        xo = zc
        # xo = torch.concat([zc, relative_dist], 1)
        error_all = torch.mean(loss)
        error.append(error_all)
        # var_list.append([self.w6, self.b6, self.w9, self.b9])
        # error = error_all + error_1 + error_2 + error_3 + error_4 + error_5 + error_6 + error_7
        return xo, error, var_list, l2_reg, reg


if __name__ == '__main__':
    print('test autoencoder')

