import torch
from torch import nn, Tensor
import torch.nn as nn
from torch.autograd import Function
import einops
import random
from typing import Tuple

import os
import sys
import warnings
import numpy as np

def get_objective(objective, batch_size, n_augs, device="cuda", fixed_waves_size = True, chop_variable_waves_size_into_segments = False, frame_separation = 1, mmcr_implicit = False, local_compression_metric="nuclear_norm"):
    # === Pretraining Loss === #
    if objective == "SimCLR":
        tau = 0.5
        return SimCLR_Loss(tau, 
                           batch_size, 
                           n_augs, 
                           device=device, 
                           fixed_waves_size=fixed_waves_size, 
                           chop_variable_waves_size_into_segments=chop_variable_waves_size_into_segments,
                           frame_separation=frame_separation)

    elif objective == "MMCR":
        lmbda=0.1
        return MMCR_Loss(lmbda, n_augs, mmcr_implicit, local_compression_metric, device)
    # === Test Loss === #
    elif objective == "CrossEntropy":
        return nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

class MMCR_Loss(nn.Module):
    def __init__(self, lmbda: float, n_aug: int, mmcr_implicit: bool, local_compression_metric: str, device: str):
        super(MMCR_Loss, self).__init__()
        self.lmbda = lmbda
        self.n_aug = n_aug
        self.mmcr_implicit = mmcr_implicit
        self.local_compression_metric = local_compression_metric
        self.device = device

        # Implicit vs. Explicit MMCR
        if self.mmcr_implicit:
            print("using Implicit MMCR")
        else:
            print("using Explicit MMCR")
            print(f"using {self.local_compression_metric}")

    def nuclear_norm(self, mat):
        try:
            S = torch.linalg.svdvals(mat)
        except Exception as e:
            print(f"SVD error | mat.shape = {mat.shape}")
            print(f"SVD error | mat = {mat}")
        return torch.sum(S)
    
    def compute_mask(self, sim_matrix, window=2):
        mask_size, _ = sim_matrix.shape
        is_odd = True if ((mask_size % 2) != 0) else False

        pos_mask = torch.zeros((mask_size,mask_size))
        ones_mat = torch.ones((window,window))
        
        # compute mask based on temporal window
        for i in range(0, mask_size, window):
            if is_odd and i == (mask_size-1):
                pass
            else:
                pos_mask[i:i+window,i:i+window] = ones_mat
        
        # set diagonals to zeros
        for j in range(mask_size):
            pos_mask[j,j] = 0

        pos_mask = pos_mask.to(self.device)

        return pos_mask

    def forward(self, z: Tensor, curr_batch_wave_labels=None) -> Tuple[Tensor, dict]:
        assert (not torch.isnan(z).any())
        num_waves = len(curr_batch_wave_labels)
        cumsum_curr_batch_wave_labels = np.cumsum(curr_batch_wave_labels)

        lst_of_centroids = []
        local_term = torch.tensor(0).float().cuda()

        for i, wave_label in enumerate(cumsum_curr_batch_wave_labels):
            if i == 0:
                curr_i_retinal_wave = z[0:wave_label,:]
            else:
                curr_i_retinal_wave = z[cumsum_curr_batch_wave_labels[i-1]:wave_label,:]

            # collect local nuclear norm
            if not self.mmcr_implicit:
                if self.local_compression_metric == "nuclear_norm":
                    local_term_i = self.nuclear_norm(curr_i_retinal_wave)
                elif self.local_compression_metric == "dot_product":
                    sim_matrix = curr_i_retinal_wave @ curr_i_retinal_wave.T
                    pos_mask = self.compute_mask(sim_matrix)
                    pos_sim = torch.sum(sim_matrix * pos_mask, dim=1)
                    local_term_i = -torch.mean(pos_sim)                
    
                local_term += local_term_i

            # collect retinal wave centroids
            curr_i_retinal_wave_centroid = torch.mean(curr_i_retinal_wave, dim=0)
            lst_of_centroids.append(curr_i_retinal_wave_centroid)
        
        # local nuclear norm
        local_term = local_term / num_waves

        # global nuclear norm
        centroids_matrix = torch.stack(lst_of_centroids,dim=0)
        global_nuc = self.nuclear_norm(centroids_matrix)

        # compute MMCR loss
        loss = -1.0 * global_nuc + self.lmbda * local_term
        loss_dict = {
            "loss": loss.item(),
            "local_term": local_term.item(),
            "global_nuc": global_nuc.item(),
        }

        return loss, loss_dict

# Reference: https://github.com/facebookresearch/vissl/blob/main/vissl/losses/simclr_info_nce_loss.py
class SimCLR_Loss(nn.Module):
    def __init__(self, 
        tau: float, 
        batch_size: int, 
        n_augs: int, 
        device: str = "cuda",

        fixed_waves_size: bool = True,
        chop_variable_waves_size_into_segments: bool = False,
        frame_separation: int = 1,
        # max_segment_distance: int=29 #based on butts paper
    ):
        super(SimCLR_Loss, self).__init__()
        self.tau = tau
        self.batch_size = batch_size
        self.n_augs = n_augs
        self.device = device
        self.fixed_waves_size = fixed_waves_size
        self.chop_variable_waves_size_into_segments = chop_variable_waves_size_into_segments
        self.frame_separation = frame_separation
        # self.max_segment_distance = max_segment_distance #push consecutive frames together

        # initialize pos_mask and neg_mask
        self.pos_mask = None
        self.neg_mask = None
        
        if self.fixed_waves_size:
            # initialize fixed pos_mask and neg_mask
            self.pos_mask = torch.zeros(self.batch_size*self.n_augs, self.batch_size*self.n_augs)
            self.neg_mask = torch.ones(self.batch_size*self.n_augs, self.batch_size*self.n_augs)
            self.precompute_mask()
        
    def precompute_mask(self):
        total_size = self.batch_size * self.n_augs
        ones_matrix = torch.ones(self.n_augs, self.n_augs)

        # create blocks of one-matrices
        for i in range(0,total_size,self.n_augs):
            self.pos_mask[i:i+self.n_augs,i:i+self.n_augs] = ones_matrix

        # set diagonals to zeros
        for j in range(total_size):
            self.pos_mask[j,j] = 0
            self.neg_mask[j,j] = 0

        # set fixed mask indices to CUDA
        self.pos_mask = self.pos_mask.to(self.device)
        self.neg_mask = self.neg_mask.to(self.device)

    def compute_dynamic_mask(self, sim_matrix, curr_batch_wave_labels):
        # initialize dynamic pos_mask and neg_mask
        self.pos_mask = torch.zeros(sim_matrix.shape)
        self.neg_mask = torch.ones(sim_matrix.shape)
        
        # compute cumulative wave manifold size
        cumsum_curr_batch_wave_labels = np.cumsum(curr_batch_wave_labels)
        
        if self.chop_variable_waves_size_into_segments:
            for i, wave_label in enumerate(cumsum_curr_batch_wave_labels):
                # negative mask
                if i == 0:
                    self.neg_mask[0:wave_label,0:wave_label] = torch.zeros(wave_label,wave_label)
                else:
                    self.neg_mask[cumsum_curr_batch_wave_labels[i-1]:wave_label,cumsum_curr_batch_wave_labels[i-1]:wave_label] = torch.zeros(curr_batch_wave_labels[i],curr_batch_wave_labels[i])
            
                # positive mask
                if i == 0:
                    for j in range(0, curr_batch_wave_labels[i]):
                        if (j+self.frame_separation)<curr_batch_wave_labels[i]:
                            compressor_matrix = torch.zeros(self.frame_separation+1,self.frame_separation+1)
                            compressor_matrix[self.frame_separation,0] = 1
                            compressor_matrix[0, self.frame_separation] = 1
                            self.pos_mask[j:j+self.frame_separation+1,j:j+self.frame_separation+1]= compressor_matrix 
                else:
                    for j in range(cumsum_curr_batch_wave_labels[i-1], cumsum_curr_batch_wave_labels[i]):
                        if (j+self.frame_separation)<cumsum_curr_batch_wave_labels[i]:
                            compressor_matrix = torch.zeros(self.frame_separation+1,self.frame_separation+1)
                            compressor_matrix[self.frame_separation,0] = 1
                            compressor_matrix[0, self.frame_separation] = 1
                            self.pos_mask[j:j+self.frame_separation+1,j:j+self.frame_separation+1]= compressor_matrix
                            
                            
            # set pos_mask diagonals to zeros
            for k in range(self.pos_mask.shape[0]):
                self.pos_mask[k,k] = 0
        else:
            for i, wave_label in enumerate(cumsum_curr_batch_wave_labels):
                if i == 0:
                    self.pos_mask[0:wave_label,0:wave_label] = torch.ones(wave_label,wave_label)
                else:
                    self.pos_mask[cumsum_curr_batch_wave_labels[i-1]:wave_label,cumsum_curr_batch_wave_labels[i-1]:wave_label] = torch.ones(curr_batch_wave_labels[i],curr_batch_wave_labels[i])

            # set diagonals to zeros
            for j in range(self.pos_mask.shape[0]):
                self.pos_mask[j,j] = 0
                self.neg_mask[j,j] = 0

        self.pos_mask = self.pos_mask.to(self.device)
        self.neg_mask = self.neg_mask.to(self.device)

    def forward(self, z: Tensor, curr_batch_wave_labels=None) -> Tuple[Tensor, dict]:
        '''Args: 
            z.shape = [self.batch_size * self.n_augs, feature_dimensions]
        '''
        # compute similarity matrix
        sim_matrix = torch.exp(torch.mm(z, z.t().contiguous()) / self.tau)

        if self.fixed_waves_size:
            pass
        else:
            self.compute_dynamic_mask(sim_matrix, curr_batch_wave_labels)
        
        pos_sim = torch.sum(sim_matrix * self.pos_mask, dim=1)
        neg_sim = torch.sum(sim_matrix * self.neg_mask, dim=1)

        # compute SimCLR loss
        pos_div_neg = pos_sim / neg_sim
        if (not self.fixed_waves_size) and self.chop_variable_waves_size_into_segments:
            pos_div_neg = pos_div_neg[pos_div_neg.nonzero(as_tuple=True)]

        loss = -(torch.mean(torch.log(pos_div_neg)))
        loss_dict = {
            "loss": loss.item(),
        }

        return loss, loss_dict
