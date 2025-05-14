from __future__ import division
import numpy as np
import torch
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
import scipy.io as sio


import model.components.gmm_variants.gmm_estimation_net_raw as dgmmb_multi
import model.components.pretrain_autoencoder as ae

torch.autograd.set_detect_anomaly(True)


def get_key(item):
    return item[0]

class PaeGmm(torch.nn.Module):
    def __init__(self, nclu_row, nclu_col, ae_config, ae_col_config, gmm_config, num_dropout, device):
        super(PaeGmm,self).__init__()
        self.autoencoder = ae.PretrainAutoencoder(ae_config, num_dropout)
        self.autoencoder_col = ae.PretrainAutoencoder(ae_col_config, num_dropout)
        self.e_net = dgmmb_multi.GMMEstimationNetRaw(gmm_config, device)
        self.e_net_col = dgmmb_multi.GMMEstimationNetRaw(gmm_config, device)
        self.gmm_optimizer = torch.optim.Adam(self.e_net.wi + self.e_net.bi + self.e_net_col.wi + self.e_net_col.bi,1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.gmm_optimizer,1500,0.1)
        self.device = device

        self.nclu_row = nclu_row # the cluster num of rows
        self.nclu_col = nclu_col

    @staticmethod
    def gaussian_normalization(train_x):
        mu = np.mean(train_x, axis=0)
        dev = np.std(train_x, axis=0)
        norm_x = (train_x - mu) / (dev + 1e-12)
        # print norm_x
        return norm_x

    @staticmethod
    def minmax_normalization(x, base):
        min_val = np.min(base, axis=0)
        max_val = np.max(base, axis=0)
        norm_x = (x - min_val) / (max_val - min_val + 1e-12)
        # print norm_x
        return norm_x

    @staticmethod
    def output_code(train_out, num_test_points):
        fout = open('../result/kddcup_output_batch_1.csv', 'w')
        for i in range(num_test_points):
            fout.write(str(train_out[i, 0]) + ',' + str(train_out[i, 1]) + ',' + str(train_out[i, 2]) + ','
                       + str(train_out[i, 3]) + '\n')
            # fout.write(str(train_out[i, 0]) + ',' + str(train_out[i, 1]) + ',' + str(train_out[i, 2]) + '\n')
        fout.close()

    def eval(self, tru, pre):
        # true label: numpy, vector in col
        # pred lable: numpy, vector in row

        num_labels = tru.shape[0]
        # nmi
        nmi = normalized_mutual_info_score(tru.reshape(num_labels), pre)

        # accuracy
        tru = np.reshape(tru, (num_labels))
        #set_tru = set(tru.tolist())
        set_pre = set(pre.tolist())
        #nclu_tru = len(set_tru) # in case that nclu_tru != the preset cluster num
        nclu_pre = len(set_pre)
        
        correct = 0
        for i in range(nclu_pre):
            flag = list(set_pre)[i]
            idx = np.argwhere(pre == flag)
            correct += max(np.bincount(tru[idx].T[0].astype(np.int64)))
        acc = correct / num_labels

        return acc, nmi

    def Reduce_Table(self, T, V1, V2):
        # T: a table (instances of the same cluster are adjacent)
        # V1: the cluster assignment vector for rows
        # V2: the cluster assignment vector for cols
        # return the reduced table

        # first reduce by rows 
        for i in range(self.nclu_row):
            T_r_data = T[V1 == i]
            if i == 0:
                T_r = torch.sum(T_r_data, dim=0)
                T_r = torch.unsqueeze(T_r,dim=0)
            else:
                temp = torch.unsqueeze(torch.sum(T_r_data, dim=0),dim=0)
                T_r = torch.concat((T_r, temp),dim=0)

        # second reduce by cols
        for i in range(self.nclu_col):
            T_rr_data = (T_r.T[V2 == i]).T
            if i == 0:
                T_rr = torch.sum(T_rr_data, dim=1)
                T_rr = torch.unsqueeze(T_rr,dim=1)
            else:
                temp = torch.unsqueeze(torch.sum(T_rr_data, dim=1),dim=1)
                T_rr = torch.concat((T_rr, temp),dim=1) 

        return T_rr

    def MI_Table(self, T):
        # T: a probablity table; numpy array
        # return the mutual information between rows and cols of table T

        P_x, P_y = torch.sum(T, 1), torch.sum(T, 0)
        nx, ny = T.shape
        T_xy = torch.matmul(torch.reshape(P_x, (nx,1)), torch.reshape(P_y, (1, ny)))

        MI_temp =torch.log((T+0.1**15)/(T_xy+0.1**15)) /torch.log(torch.Tensor([2]).to(self.device))
        MI_T = torch.sum(torch.multiply(T, MI_temp))

        return MI_T


    def MI_loss(self, Ur, Uc):
        # Ur: the cluster assignment matrix for rows, N_ins * N_Row_clus
        # Uc: the cluster assignment matrix for cols, N_att * N_Col_clus
        # return the loss of mutual information between the original data matrix and the reduced data matrix

        N_ins, N_Row_clus = Ur.shape
        N_att, N_Col_clus = Uc.shape
        #T_pro_org = torch.zeros((N_ins, N_att))) # original table for the joint probability
        #T_pro_red = torch.zeros((N_Row_clus, N_Col_clus))) # reduced table for the joint probability
        Ur_max = torch.reshape(torch.max(Ur, dim=1).values, (N_ins, 1)) # the max value for each row; column vector
        Uc_max = torch.reshape(torch.max(Uc, dim=1).values, (N_att, 1))

        Ur_max_idx = torch.argmax(Ur, dim=1) # the index of max value for rows, vector in row
        Uc_max_idx = torch.argmax(Uc, dim=1)

        # T_pro_org = torch.matmul(Ur_max, Uc.T) # original table for the joint probability
        T_pro_org = torch.matmul(Ur, Uc.T) # original table for the joint probability
        T_pro_org = T_pro_org / torch.sum(T_pro_org) # normalization: make sum equal 1

        T_pro_red = self.Reduce_Table(T_pro_org, Ur_max_idx, Uc_max_idx) # reduced table
        
        # calculate MI of orginal and reduced tables
        MI_org = self.MI_Table(T_pro_org)
        MI_red = self.MI_Table(T_pro_red)
        
        #loss = torch.abs((1 - MI_red/MI_org)*(MI_org - MI_red)) # alternative calculation way
        loss = torch.abs(1 - MI_red/MI_org)
        loss =torch.log(1+loss)


        self.T_pro_org = T_pro_org
        self.sum_T_pro_org = torch.sum(T_pro_org)
        self.T_pro_red = T_pro_red
        self.sum_T_pro_red = torch.sum(T_pro_red)
        self.Ur = Ur
        self.Uc = Uc
        self.Ur_max = Ur_max
        self.Uc_max = Uc_max
        self.Ur_max_idx = Ur_max_idx
        self.Uc_max_idx = Uc_max_idx
        self.Ur_max_idx_sum = torch.sum(Ur_max_idx)
        self.Uc_max_idx_sum = torch.sum(Uc_max_idx)
        self.MI_org = MI_org
        self.MI_red = MI_red

        return loss

    def run(self, data, labels, train_epochs, pretrain_epochs):
        # Data
        train_x = data
        train_y = labels
        train_x_col = data.T

        # train_norm_x = self.minmax_normalization(train_x, train_x)
        # train_norm_x_col = self.minmax_normalization(train_x_col, train_x_col)

        # Setup
        train_x_v = torch.from_numpy(self.gaussian_normalization(train_x)).to(torch.float32).to(self.device)
        train_x_v_col = torch.from_numpy(self.gaussian_normalization(train_x_col)).to(torch.float32).to(self.device)
        keep_prob = 1.0 

        RR_acc = []
        RR_nmi = []
    
        # Pretraining
        train_z, train_error, train_l2_reg = self.autoencoder.fit(train_x_v,pretrain_epochs,keep_prob) 
        train_z_col, train_error_col, train_l2_reg_col = self.autoencoder_col.fit(train_x_v_col,pretrain_epochs,keep_prob) 

        # Joint fine training
        error_oa = 0
        for error_k in train_error:
            error_oa = error_oa + error_k
        # error_oa = train_error[len(train_error) - 1]
        # reconstruction_error = train_error[len(train_error) - 1]

        error_oa_col = 0
        for error_col_k in train_error_col:
            error_oa_col = error_oa_col + error_col_k

        # GMM Membership estimation
        for epoch in range(train_epochs):
            self.train()

            loss, pen_dev, likelihood, p_z, x_t, p_t, z_p, z_t, mixture_mean, mixture_dev, mixture_cov, mixture_dev_det = self.e_net.run(train_z, keep_prob)

            loss_col, pen_dev_col, likelihood_col, p_z_col, x_t_col, p_t_col, z_p_col, z_t_col, mixture_mean_col, mixture_dev_col, mixture_cov_col, mixture_dev_det_col = self.e_net_col.run(train_z_col, keep_prob)

            # Train step

            # Para set for Softmax(h) (h is last representation) and Softmax(h)
            # obj_oa = error_oa * 1e1 + train_l2_reg * 1e-2 + loss * 1e1 + pen_dev * 1e-2
            # obj_oa_row = error_oa * 1e0 + train_l2_reg * 5e-1 + loss * 5e0 + pen_dev * 1e1

            # Para set for h (last representation) and Softmax(h)
            # obj_oa_row =     error_oa * 5e1 +     train_l2_reg * 1e1 +     loss * 5e0 +     pen_dev * 5e1

            obj_oa_row =     error_oa * 2e-2  +     train_l2_reg * 2e-2 +     loss * 1e-1 +     pen_dev
            obj_oa_col = error_oa_col * 2e-2  + train_l2_reg_col * 2e-2 + loss_col * 1e-1 + pen_dev_col
            obj_cross  = self.MI_loss(p_z, p_z_col)
            obj_oa     = obj_oa_row + obj_oa_col + obj_cross * 1e5

            self.gmm_optimizer.zero_grad()
            obj_oa.requires_grad = True
            obj_oa.backward()
            self.gmm_optimizer.step()

            self.scheduler.step()

            if epoch % 20 == 0 : print(f'epoch {epoch}, loss : {obj_oa}')
            # calculate accuracy and NMI
            pred_label = np.argmax(p_z.cpu(), 1) # vertor in row
            pred_label_col = np.argmax(p_z_col.cpu(), 1)

            true_label = train_y # numpy
            acc, NMI = self.eval(true_label, pred_label)
            #print('acc:' + str(acc) + ',' + 'NMI:' + str(NMI))

            RR_acc = np.append(RR_acc, acc)
            RR_nmi = np.append(RR_nmi, NMI)
            

        return RR_acc, RR_nmi

