# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 11:37:52 2023

@author: xzhou
"""
import torch
import torch.nn.functional as F

def multi_contrastive_loss(embP, embM, embF,PP_adj,MM_adj,PM_adj,PF_adj,MF_adj,tau):
    loss = torch.tensor(0, dtype=float, requires_grad=True)    
    loss = loss + P_con_loss(embP, embM, embF,PM_adj,PP_adj,PF_adj, tau=tau)
    loss = loss + M_con_loss(embP, embM, embF,PM_adj.T,MM_adj,MF_adj, tau=tau)    
    loss = loss + F_con_loss(embP, embM, embF,MF_adj.T,PF_adj.T, tau=tau)   
    return loss/3 

def sim(z1: torch.Tensor, z2: torch.Tensor, hidden_norm: bool = True):
    if hidden_norm:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())
      

def P_con_loss(embP: torch.Tensor, embM: torch.Tensor,embF: torch.Tensor, PM_adj: torch.Tensor,PP_adj: torch.Tensor,PF_adj: torch.Tensor, 
                         tau,hidden_norm: bool = True):
    nei_count = torch.sum(PM_adj, 1) +torch.sum(PP_adj, 1) +torch.sum(PF_adj, 1) # intra-view nei+inter-view nei+self inter-view
    nei_count = torch.squeeze(torch.tensor(nei_count))

    f = lambda x: torch.exp(x / tau)
    PM_sim = f(sim(embP, embM, hidden_norm))
    PP_sim = f(sim(embP, embP, hidden_norm))
    PF_sim = f(sim(embP, embF, hidden_norm))
    
    loss = ((PM_sim.mul(PM_adj)).sum(1) + (PP_sim.mul(PP_adj)).sum(1)+(PF_sim.mul(PF_adj)).sum(1)) / (
            (PM_sim.mul(PM_adj)).sum(1)+ PP_sim.sum(1)+PF_sim.sum(1))
    nei_count[nei_count==0]=1
    loss = loss / nei_count  # divided by the number of positive pairs for each node
    loss[loss==0]=1

    return (-torch.log(loss)).mean()

def M_con_loss(embP: torch.Tensor, embM: torch.Tensor,embF: torch.Tensor, MP_adj: torch.Tensor,MM_adj: torch.Tensor,MF_adj: torch.Tensor, 
                         tau,hidden_norm: bool = True):
    nei_count = torch.sum(MP_adj, 1) +torch.sum(MM_adj, 1) +torch.sum(MF_adj, 1) # intra-view nei+inter-view nei+self inter-view
    nei_count = torch.squeeze(torch.tensor(nei_count))

    f = lambda x: torch.exp(x / tau)
    MP_sim = f(sim(embM, embP, hidden_norm))
    MM_sim = f(sim(embM, embM, hidden_norm))
    MF_sim = f(sim(embM, embF, hidden_norm))
    
    loss = ((MP_sim.mul(MP_adj)).sum(1) + (MM_sim.mul(MM_adj)).sum(1)+(MF_sim.mul(MF_adj)).sum(1)) / (
            (MP_sim.mul(MP_adj)).sum(1)+ MM_sim.sum(1)+MF_sim.sum(1))
    nei_count[nei_count==0]=1
    loss = loss / nei_count  # divided by the number of positive pairs for each node
    loss[loss==0]=1

    return (-torch.log(loss)).mean()

def F_con_loss(embP: torch.Tensor, embM: torch.Tensor,embF: torch.Tensor, FM_adj: torch.Tensor,FP_adj: torch.Tensor, 
                         tau,hidden_norm: bool = True):
    nei_count = torch.sum(FM_adj, 1) +torch.sum(FP_adj, 1) 
    nei_count = torch.squeeze(torch.tensor(nei_count))    
    f = lambda x: torch.exp(x / tau)
    FM_sim = f(sim(embF, embM, hidden_norm))
    FP_sim = f(sim(embF, embP, hidden_norm))    
    loss = ((FM_sim.mul(FM_adj)).sum(1)+(FP_sim.mul(FP_adj)).sum(1)) /(FM_sim.sum(1)+FP_sim.sum(1))    
    nei_count[nei_count==0]=1
    loss = loss / nei_count    
    loss[loss==0]=1

    return (-torch.log(loss)).mean()  
