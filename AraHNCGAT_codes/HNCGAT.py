# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 22:46:44 2023

@author: xzhou
"""
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from scipy import sparse
from utils import readListfile, get_train_index, calculateauc
import argparse
import warnings
from loss import multi_contrastive_loss
from datetime import date

# protein: nodeA; Metabolite: nodeB; GO: nodeC

class MPINet(torch.nn.Module):
    def __init__(self, nodeA_num, nodeB_num, nodeA_feature_num, nodeB_feature_num, nodeC_feature_num, hidden_dim,
                 dropout,edgetype):
        super(MPINet, self).__init__()
        self.encoder_1 = graphNetworkEmbbed(nodeA_num, nodeB_num, nodeA_feature_num, nodeB_feature_num,
                                            nodeC_feature_num, hidden_dim, dropout)
        self.decoder = MlpDecoder(hidden_dim,edgetype)

    def forward(self, data_set, adj_AB, adj_AC, adj_BC, nodeA_feature, nodeB_feature, nodeC_feature, adj_A_sim,
                adj_B_sim,edgetype):
        nodeA_feature, nodeB_feature, nodeC_feature = self.encoder_1(adj_AB, adj_AC, adj_BC,
                                                                     nodeA_feature, nodeB_feature, nodeC_feature,
                                                                     adj_A_sim, adj_B_sim)

        predictAfeature = nodeA_feature[data_set[:, 0],]
        predictBfeature = nodeB_feature[data_set[:, 1],]

        prediction = self.decoder(predictAfeature, predictBfeature,edgetype).flatten()
        return prediction,nodeA_feature, nodeB_feature, nodeC_feature


class biattenlayer(torch.nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(biattenlayer, self).__init__()
        self.atten_ItoJ = nn.Conv1d(hidden_dim, 1, 1)
        self.atten_JtoI = nn.Conv1d(hidden_dim, 1, 1)
        self.dropout = dropout
        self.reset_parameters()
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.02)
        self.softmax = torch.nn.Softmax(dim=1)

    def reset_parameters(self):
        
        nn.init.xavier_normal_(self.atten_ItoJ.weight, gain=1.414)
        nn.init.xavier_normal_(self.atten_JtoI.weight, gain=1.414)

    def forward(self, nodeI_mlp, nodeJ_mlp, adj_mat):
        """
        return 0 if adj>0, -1e9 if adj=0
        """
        X_I = torch.unsqueeze(torch.transpose(nodeI_mlp, 0, 1), 0) #shape (1,hid,num_I)
        X_J = torch.unsqueeze(torch.transpose(nodeJ_mlp, 0, 1), 0) #shape (1,hid,num_J)
        ### a*[h_i||h_j]=a_i*h_i + a_j*h_j
        f_ItoJ = self.atten_ItoJ(X_I)  # shape (1,1,num_I), a_i*h_i
        f_JtoI = self.atten_JtoI(X_J)  # shape (1,1,num_J), a_j*h_j
        edge_logits = f_ItoJ + torch.transpose(f_JtoI, 2, 1)  # shape (1,num_J,num_I)
        
        edge_logits = torch.squeeze(edge_logits)  # from (1,num_J,num_I) to (num_J,num_I)
        ###bias mat is 0 if adj>0, -1e9 if adj=0
        edge_logits = self.leakyrelu(edge_logits)
        edge_logits=edge_logits.T # from (num_J,num_I) to (num_I,num_J)
        ###softmax only among neighborhood
        zero_vec = -1e9 * torch.ones_like(edge_logits)
        ##if adj>0, keep edge_logits; if adj=0, replace by -1e9
        
        att_IJ = torch.where(adj_mat > 0, edge_logits, zero_vec)
        att_IJ = self.softmax(att_IJ)
        rowmean=1/att_IJ.size()[1] 
        att_IJ = torch.where(abs(att_IJ-rowmean)<1e-9,torch.tensor(0.).cuda(),att_IJ)        
        return att_IJ


class selfattenlayer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(selfattenlayer, self).__init__()
        self.atten_ItoJ = nn.Conv1d(hidden_dim, 1, 1)
        self.atten_JtoI = nn.Conv1d(hidden_dim, 1, 1)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.02)
        self.softmax = torch.nn.Softmax(dim=1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.atten_ItoJ.weight, gain=1.414)
        nn.init.xavier_normal_(self.atten_JtoI.weight, gain=1.414)

    def forward(self, node_mlp, adj_mat):
        X_IJ = torch.unsqueeze(torch.transpose(node_mlp, 0, 1), 0)  # shape  (1,n_emb,num_nodes)
        ### a*[h_i||h_j]=a_i*h_i + a_j*h_j
        f_ItoJ = self.atten_ItoJ(X_IJ)  # shape (1,1,num_nodes), a_i*h_i
        f_JtoI = self.atten_JtoI(X_IJ)  # shape (1,1,num_nodes), a_j*h_j
        edge_logits = f_ItoJ + torch.transpose(f_JtoI, 2, 1)  # shape (1,num_nodes,num_nodes)
        edge_logits = torch.squeeze(edge_logits)  # from (1,num_nodes,num_nodes) to (num_nodes,num_nodes)
        ###bias mat is 0 if adj>0, -1e9 if adj=0
        edge_logits = self.leakyrelu(edge_logits)
        ###softmax only among neighborhood
        zero_vec = -1e9 * torch.ones_like(edge_logits)
        ##if adj>0, keep edge_logits; if adj=0, replace by -1e9
        att_IJ = torch.where(adj_mat > 0, edge_logits, zero_vec)
        att_IJ = self.softmax(att_IJ)
        return att_IJ


class graphNetworkEmbbed(torch.nn.Module):
    def __init__(self, nodeA_num, nodeB_num, nodeA_feature_num, nodeB_feature_num, nodeC_feature_num, hidden_dim,
                 dropout):
        super(graphNetworkEmbbed, self).__init__()

        self.mlp_A = nn.Linear(nodeA_feature_num, hidden_dim)
        self.mlp_B = nn.Linear(nodeB_feature_num, hidden_dim)
        self.mlp_C = nn.Linear(nodeC_feature_num, hidden_dim)

        self.attAB_adj = biattenlayer(hidden_dim, dropout=dropout)
        self.attBA_adj = biattenlayer(hidden_dim, dropout=dropout)
        self.attBC_adj = biattenlayer(hidden_dim, dropout=dropout)
        self.attCB_adj = biattenlayer(hidden_dim, dropout=dropout)
        self.attAC_adj = biattenlayer(hidden_dim, dropout=dropout)
        self.attCA_adj = biattenlayer(hidden_dim, dropout=dropout)
        
        self.emblayerAB = nn.Linear(2 * hidden_dim, hidden_dim)
        self.emblayerBA = nn.Linear(2 * hidden_dim, hidden_dim)
        
        self.emblayerBC = nn.Linear(2 * hidden_dim, hidden_dim)
        self.emblayerCB = nn.Linear(2 * hidden_dim, hidden_dim)
        
        self.emblayerAC = nn.Linear(2 * hidden_dim, hidden_dim)
        self.emblayerCA = nn.Linear(2 * hidden_dim, hidden_dim)

        self.mlp_sim_A = nn.Linear(nodeA_num, hidden_dim)
        self.mlp_sim_B = nn.Linear(nodeB_num, hidden_dim)

        self.attA_sim = selfattenlayer(nodeA_feature_num, hidden_dim, dropout=dropout)
        self.attB_sim = selfattenlayer(nodeB_feature_num, hidden_dim, dropout=dropout)
        self.nodeA_emb = nn.Linear(hidden_dim * 4, hidden_dim)
        self.nodeB_emb = nn.Linear(hidden_dim * 4, hidden_dim)
        self.nodeC_emb = nn.Linear(hidden_dim * 3, hidden_dim)

        self.dropout = dropout
        self.reset_parameters()   

    def reset_parameters(self):
        nn.init.xavier_normal_(self.mlp_A.weight, gain=1.414)
        nn.init.xavier_normal_(self.mlp_B.weight, gain=1.414)
        nn.init.xavier_normal_(self.mlp_C.weight, gain=1.414)

        nn.init.xavier_normal_(self.emblayerAB.weight, gain=1.414)
        nn.init.xavier_normal_(self.emblayerBA.weight, gain=1.414)

        nn.init.xavier_normal_(self.emblayerBC.weight, gain=1.414)
        nn.init.xavier_normal_(self.emblayerCB.weight, gain=1.414)

        nn.init.xavier_normal_(self.emblayerAC.weight, gain=1.414)
        nn.init.xavier_normal_(self.emblayerCA.weight, gain=1.414)

        nn.init.xavier_normal_(self.mlp_sim_A.weight, gain=1.414)
        nn.init.xavier_normal_(self.mlp_sim_B.weight, gain=1.414)

        nn.init.xavier_normal_(self.nodeA_emb.weight, gain=1.414)
        nn.init.xavier_normal_(self.nodeB_emb.weight, gain=1.414)
        nn.init.xavier_normal_(self.nodeC_emb.weight, gain=1.414)

    def forward(self, adj_AB, adj_AC, adj_BC, nodeA_feature, nodeB_feature, nodeC_feature,
                adj_A_sim, adj_B_sim):
        nodeA_mlp = F.relu(self.mlp_A(nodeA_feature))
        nodeA_mlp = F.dropout(nodeA_mlp, self.dropout, training=self.training)
        nodeB_mlp = F.relu(self.mlp_B(nodeB_feature))
        nodeB_mlp = F.dropout(nodeB_mlp, self.dropout, training=self.training)
        nodeC_mlp = F.relu(self.mlp_C(nodeC_feature))
        nodeC_mlp = F.dropout(nodeC_mlp, self.dropout, training=self.training)
        
        nodeA_feature_from_sim = F.relu(self.mlp_sim_A(adj_A_sim)) # biasmat(adj_A_sim)
        nodeB_feature_from_sim = F.relu(self.mlp_sim_B(adj_B_sim))

        att_AB = self.attAB_adj(nodeA_mlp, nodeB_mlp, adj_AB)
        att_BA = self.attBA_adj(nodeB_mlp, nodeA_mlp, adj_AB.T)

        nodeA_feature_from_nodeB = F.relu(self.emblayerAB(torch.cat((nodeA_mlp,torch.mm(att_AB, nodeB_mlp)),1)))

        nodeB_feature_from_nodeA = F.relu(self.emblayerBA(torch.cat((nodeB_mlp,torch.mm(att_BA, nodeA_mlp)),1)))

        att_BC = self.attBC_adj(nodeB_mlp, nodeC_mlp, adj_BC)
        att_CB = self.attCB_adj(nodeC_mlp, nodeB_mlp, adj_BC.T)
        nodeB_feature_from_nodeC = F.relu(self.emblayerBC(torch.cat((nodeB_mlp,torch.mm(att_BC, nodeC_mlp)),1)))
        nodeC_feature_from_nodeB = F.relu(self.emblayerCB(torch.cat((nodeC_mlp,torch.mm(att_CB, nodeB_mlp)),1)))
        
        att_AC = self.attAC_adj(nodeA_mlp, nodeC_mlp, adj_AC)
        att_CA = self.attCA_adj(nodeC_mlp, nodeA_mlp, adj_AC.T)

        nodeA_feature_from_nodeC = F.relu(self.emblayerBC(torch.cat((nodeA_mlp,torch.mm(att_AC, nodeC_mlp)),1)))
        nodeC_feature_from_nodeA = F.relu(self.emblayerCB(torch.cat((nodeC_mlp,torch.mm(att_CA, nodeA_mlp)),1)))
        
        nodeA_emb = F.relu(
            self.nodeA_emb(torch.cat((nodeA_feature_from_sim, nodeA_feature_from_nodeB, nodeA_feature_from_nodeC, nodeA_mlp), 1)))# 230808 add nodeA_mlp
        nodeB_emb = F.relu(
            self.nodeB_emb(torch.cat((nodeB_feature_from_sim, nodeB_feature_from_nodeC, nodeB_feature_from_nodeA, nodeB_mlp), 1)))

        nodeC_emb = F.relu(self.nodeC_emb(torch.cat((nodeC_feature_from_nodeA, nodeC_feature_from_nodeB, nodeC_mlp), 1)))

        return nodeA_emb, nodeB_emb, nodeC_emb


class MlpDecoder(torch.nn.Module):

    def __init__(self, input_dim,edgetype):
        super(MlpDecoder, self).__init__()
        if edgetype=='concat':
            self.mlp_1 = nn.Sequential(nn.Dropout2d(p=0.1),
                                   nn.Linear(int(input_dim * 2), int(input_dim)),
                                   nn.ReLU())
        else:
            self.mlp_1 = nn.Sequential(nn.Dropout2d(p=0.1),
                                   nn.Linear(int(input_dim), int(input_dim)),
                                   nn.ReLU())
        self.mlp_2 = nn.Sequential(nn.Dropout2d(p=0.1),
                                   nn.Linear(int(input_dim), int(input_dim // 2)),
                                   nn.ReLU())
        self.mlp_3 = nn.Sequential(nn.Dropout2d(p=0.1),
                                   nn.Linear(int(input_dim // 2), 1),
                                   nn.Sigmoid())

    def forward(self, nodeI_feature, nodeJ_feature,edgetype):
        if edgetype == 'concat':
            pair_feature = torch.cat([nodeI_feature, nodeJ_feature], 1)
        elif edgetype == 'L1':
            pair_feature = torch.abs(nodeI_feature- nodeJ_feature)
        elif edgetype == 'L2':
            pair_feature = torch.square(nodeI_feature- nodeJ_feature)     
        elif edgetype == 'had':
            pair_feature = torch.mul(nodeI_feature,nodeJ_feature)
        elif edgetype == 'mean':
            pair_feature = torch.add(nodeI_feature,nodeJ_feature) / 2   
            
        embedding_1 = self.mlp_1(pair_feature)
        embedding_2 = self.mlp_2(embedding_1)
        outputs = self.mlp_3(embedding_2)
        return outputs


def biasmat(adj_IJ_expand):
    mt = (adj_IJ_expand > 0) * 1
    bias_mat = -1e9 * (1.0 - mt)
    bias_mat = torch.from_numpy(bias_mat).float()
    return bias_mat


def main(args):
    torch.manual_seed(0)
    np.random.seed(0)
    # protein: nodeA; Metabolite: nodeB; GO: nodeC

    adj_B_sim = torch.Tensor(np.float16(sparse.load_npz('compoundsim.npz').todense()))
    adj_BC = np.int16(sparse.load_npz('compoundGO.npz').todense())
    adj_A_sim = torch.Tensor(np.int16(sparse.load_npz('araproteinLitePPI.npz').todense()))
    adj_AC = np.int16(sparse.load_npz('proteinLiteGO.npz').todense())
    adj_AB = np.int16(sparse.load_npz('proteinLiteCompound.npz').todense())
    
    metaboliteList = readListfile('compoundList.txt')
    proteinList = readListfile('proteinListLite.txt')
    GOList = readListfile('GOList.txt')

    f = open('./AraHNCGAT_result'+str(date.today())+'.txt', 'a')
    f.write('{}'.format(args))
    f.write('\n')
    f.flush()

    nodeB_num = len(metaboliteList)
    nodeA_num = len(proteinList)
    nodeC_num = len(GOList)

    pos_u, pos_v = np.where(adj_AB != 0)
    neg_u, neg_v = np.where(adj_AB == 0)

    negative_ratio = 10
    negative_sample_index = np.random.choice(np.arange(len(neg_u)), size=negative_ratio * len(pos_u), replace=False)

    pos_data_set = np.zeros((len(pos_u), 3), dtype=int)
    neg_data_set = np.zeros((len(negative_sample_index), 3), dtype=int)

    for i in range(len(pos_u)):
        pos_data_set[i][0] = pos_u[i]
        pos_data_set[i][1] = pos_v[i]
        pos_data_set[i][2] = 1
    count = 0
    for i in negative_sample_index:
        neg_data_set[count][0] = neg_u[i]
        neg_data_set[count][1] = neg_v[i]
        neg_data_set[count][2] = 0
        count = count + 1
    hidden_dim = args.hid_dim
    dropout = args.dropout
    lamb = args.lamb
    lr = args.lr
    edgetype=args.edgetype
    tau=args.temperature
    
    nodeA_feature_num = hidden_dim
    nodeB_feature_num = hidden_dim
    nodeC_feature_num = hidden_dim


    adj_AC=torch.Tensor(adj_AC)
    adj_BC=torch.Tensor(adj_BC)

    trp_varied = [0.9]
    AUC_ROCAll = []
    AUCstdALL = []
    AUC_PRAll = []
    AUC_PRstdALL = []
    for train_ratio in trp_varied:
        val_ratio = 0
        test_ratio = 1 - train_ratio - val_ratio
        numRandom = 5
        AUC_ROCtrp = []
        AUC_PRtrp = []                  
            
        for random_state in range(numRandom):
            print("%d-th random split with training ratio %f" % (random_state + 1, train_ratio))
            modelfilename = 'rand' + str(random_state) + 'trp' + str(train_ratio) + '_best_conMPI.pkl'                

            pos_idx_train, pos_idx_val, pos_idx_test, pos_y_train, pos_y_val, pos_y_test, pos_train_mask, pos_val_mask, pos_test_mask = get_train_index(
                pos_u, train_ratio, val_ratio, test_ratio, numRandom, random_state)
            neg_idx_train, neg_idx_val, neg_idx_test, neg_y_train, neg_y_val, neg_y_test, neg_train_mask, neg_val_mask, neg_test_mask = get_train_index(
                negative_sample_index, train_ratio, val_ratio, test_ratio, numRandom, random_state)
            train_adj_AB = np.zeros((nodeA_num, nodeB_num), dtype=int)
            for i in pos_idx_train:
                idxi = pos_data_set[i, 0]
                idxj = pos_data_set[i, 1]
                train_adj_AB[idxi][idxj] = 1

            train_mask = np.array(np.concatenate((pos_train_mask, neg_train_mask), 0), dtype=np.bool)
            train_mask = torch.tensor(train_mask)
            test_mask = np.array(np.concatenate((pos_test_mask, neg_test_mask), 0), dtype=np.bool)
            test_mask = torch.tensor(test_mask)
            data_set = np.concatenate((pos_data_set, neg_data_set), 0)
            data_set = torch.tensor(data_set).long()

            train_adj_AB=torch.Tensor(train_adj_AB)
            
            model = MPINet(nodeA_num, nodeB_num, nodeA_feature_num, nodeB_feature_num, nodeC_feature_num, hidden_dim,
                           dropout,edgetype)

            loss_func = torch.nn.BCELoss(reduction='mean')
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-7)

            test_AUC_ROC_list = []
            test_AUCpr_list = []
            test_AUC_AUCpr_best = 0
            test_AUC_best = 0
            test_AUCpr_best = 0
            best_epoch = 0
            nodeA_feature = torch.rand(nodeA_num, nodeA_feature_num, requires_grad=True)
            nodeB_feature = torch.rand(nodeB_num, nodeB_feature_num, requires_grad=True)
            nodeC_feature = torch.rand(nodeC_num, nodeC_feature_num, requires_grad=True)
            if args.gpu >= 0 and torch.cuda.is_available():
                model = model.cuda()
                data_set = data_set.cuda()

                nodeA_feature = nodeA_feature.cuda()
                nodeB_feature = nodeB_feature.cuda()
                nodeC_feature = nodeC_feature.cuda()
                adj_A_sim = adj_A_sim.cuda()
                adj_B_sim = adj_B_sim.cuda()
                adj_AC=adj_AC.cuda()
                adj_BC=adj_BC.cuda()
                train_adj_AB=train_adj_AB.cuda()                
                
            for epoch in range(args.n_epochs):

                model.train()
                opt.zero_grad()
                torch.autograd.set_detect_anomaly(True)
                prob,embA,embB,embC = model(data_set, train_adj_AB, adj_AC, adj_BC, nodeA_feature, nodeB_feature,
                             nodeC_feature, adj_A_sim, adj_B_sim,edgetype)
                label = data_set[:, 2].float()
                
                train_auc, _ = calculateauc(prob[train_mask], data_set[:, 2][train_mask])
                conloss=multi_contrastive_loss(embA,embB,embC,adj_A_sim, adj_B_sim,train_adj_AB,adj_AC,adj_BC,tau)
                
                loss = loss_func(prob[train_mask], label[train_mask])+lamb*conloss 
                loss.backward()
                opt.step()
                model.eval()
                with torch.no_grad():
                    logits,_,_,_ = model(data_set, train_adj_AB, adj_AC, adj_BC,
                                   nodeA_feature, nodeB_feature, nodeC_feature, adj_A_sim, adj_B_sim,edgetype)

                    logits = logits[test_mask]
                    label = data_set[:, 2][test_mask]
                    testAUC_ROC, testAUCpr = calculateauc(logits, label)
                    if test_AUC_AUCpr_best <= (testAUC_ROC + testAUCpr):
                        torch.save(model.state_dict(), modelfilename)
                        test_AUC_AUCpr_best = (testAUC_ROC + testAUCpr)
                        test_AUC_best = testAUC_ROC
                        test_AUCpr_best = testAUCpr
                        best_epoch = epoch

                test_AUC_ROC_list.append(testAUC_ROC)
                test_AUCpr_list.append(testAUCpr)

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print("Epoch {:03d} | Loss {:.4f} | TrainAUC {:.4f} |"
                          " testAUC_ROC {:.4f} |  testAUCpr {:.4f} ".
                          format(epoch + 1, loss.item(), train_auc,
                                 testAUC_ROC, testAUCpr))

            AUC_ROCtrp.append(test_AUC_best)
            AUC_PRtrp.append(test_AUCpr_best)

            model.load_state_dict(torch.load(modelfilename))
            model.eval()
            with torch.no_grad():
                logits,_,_,_ = model(data_set, train_adj_AB, adj_AC, adj_BC,
                               nodeA_feature, nodeB_feature, nodeC_feature, adj_A_sim, adj_B_sim,edgetype)

                logits = logits[test_mask]
                label = data_set[:, 2][test_mask]
                testAUC_ROC, testAUCpr = calculateauc(logits, label)
                print("Load model result: Epoch {:03d}"
                      " testAUC_ROC {:.4f} |  testAUCpr {:.4f} ".
                      format(best_epoch + 1, testAUC_ROC, testAUCpr))

        AUC_ROCAll.append(np.mean(AUC_ROCtrp))
        AUCstdALL.append(np.std(AUC_ROCtrp))
        AUC_PRAll.append(np.mean(AUC_PRtrp))
        AUC_PRstdALL.append(np.std(AUC_PRtrp))

        f.write('avg AUC_ROC: %f + %f, for trp:%.2f \n' % (np.mean(AUC_ROCtrp), np.std(AUC_ROCtrp), train_ratio))
        f.write('avg  AUC_PRtrp: %f + %f, for trp:%.2f \n' % (np.mean(AUC_PRtrp), np.std(AUC_PRtrp), train_ratio))

        f.flush()

    print('AUC_ROCAll: \n')
    f.write('AUC_ROCAll:')
    for auc in AUC_ROCAll:
        print('%f \n' % (auc))
        f.write('%f,' % (auc))
    f.write('\n' + 'AUCstd:')
    for aucstd in AUCstdALL:
        print('%f \n' % (aucstd))
        f.write('%f,' % (aucstd))

    f.write('\n' + 'AUC_PRAll:')
    print('AUC_PRAll: \n')
    for AUC_PR in AUC_PRAll:
        print('%f \n' % (AUC_PR))
        f.write('%f,' % (AUC_PR))
    f.write('\n' + 'AUC_PRstd:')
    for AUC_PRstd in AUC_PRstdALL:
        print('%f \n' % (AUC_PRstd))
        f.write('%f,' % (AUC_PRstd))

    f.write('\n')
    f.flush()

    f.close()


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='conMPI')

    parser.add_argument('--gpu', type=int, default=1, help='GPU index. Default: -1, using CPU.')
   
    parser.add_argument('--n-epochs', type=int, default=1000, help='Training epochs.')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate.')
    parser.add_argument('--lamb', type=float, default=1, help='conloss weight.')
   
    parser.add_argument("--hid-dim", type=int, default=64, help='Hidden layer dimensionalities.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature')
    parser.add_argument('--edgetype', type=str, default='concat', help='edgetype')
    args = parser.parse_args()
    print(args)
    warnings.filterwarnings("ignore")
    main(args)                                              
