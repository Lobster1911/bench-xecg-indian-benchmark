import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def focal_loss(loss):
    pt = torch.exp(-loss)
    alpha = 2.
    gamma = .25
    loss = (alpha * (1-pt)**gamma * loss)
    return loss.mean()   

def masked_cosine_loss(input, target, reduction='mean', mask=None):
    loss = F.cosine_similarity(target, input, dim=-1)
    if mask is not None:
        loss = loss[mask]

    if reduction == "mean":
        return 1 - loss.mean()
    elif reduction == "sum":
        return (1-loss).sum()
    else:
        return 1-loss

def masked_mse_loss(input, target, reduction='mean', mask=None):
    out = (input - target)**2
    if mask is not None:
        out = out[mask]
    if reduction == "mean":
        return out.mean()
    elif reduction == "sum":
        return out.sum()
    else:
        return out
    
def masked_mae_loss(input, target, reduction='mean', mask=None):
    out = torch.abs(input-target)
    # do not consider elements set to 0
    if mask is not None:
        #expand the mask with 12 channels
        out = out[mask]
    if reduction == "mean":
        return out.mean()
    elif reduction == "sum":
        return out.sum()
    else:
        return out
    
def masked_min_max_loss(input, target, reduction='mean', patch_size=100, mask=None):
    # input should be tokenized
    batch_size, sig_len, num_channels = input.shape
    #print('batch_size', batch_size)
    #print('sig_len', sig_len)
    tokens_num = sig_len // patch_size    
    tokenized_inp = input.view(batch_size, tokens_num, patch_size, num_channels)
    tokenized_target = target.view(batch_size, tokens_num, patch_size, num_channels)   
    # find the min and max value of each patch
    min_inp, _ = tokenized_inp.min(dim=-1)
    # print('min_inp', min_inp.shape)
    max_inp, _ = tokenized_inp.max(dim=-1)
    # print('max_inp', max_inp)
    min_target, _ = tokenized_target.min(dim=-1)
    # print('min_target', min_target)
    max_target, _ = tokenized_target.max(dim=-1)
    # print('max_target', max_target)
    # calculate the loss
    out = (min_inp - min_target)**2 + (max_inp - max_target)**2

    # do not consider elements set to 0
    if mask is not None:
        out = out[mask]

    if reduction == "mean":
        return out.mean() / tokens_num
    elif reduction == "sum":
        return out.sum() / tokens_num
    else:
        return out / tokens_num
    
def gradient_loss(input, target, reduction='mean', p=2, mask=None):
    input_grad = input[:, 1:] - input[:, :-1]
    target_grad = target[:, 1:] - target[:, :-1]
    out = torch.pow(input_grad - target_grad, p)
    # do not consider elements that was at 0 in the input
    # using [:, :-1] because preserve the order of the elements
    if mask is not None:
        #expand the mask with 12 channels
        out = out[mask[:, :-1]]
    if reduction == "mean":
        return out.mean()
    elif reduction == "sum":
        return out.sum()
    else:
        return out
    

class SimDINOv2Loss(nn.Module):
    def __init__(self, eps=0.5, coeff=1.0):
        super().__init__()
        self.eps = eps
        self.coeff = coeff

    def forward(self, student_feat, teacher_feat):
        """
        Expansion Loss and Compression Loss between features of the teacher and student networks.
        """
        # student_feat = student_feat.view(2, -1, student_feat.shape[-1])
        # teacher_feat = teacher_feat.view(2, -1, teacher_feat.shape[-1])

        student_feat = F.normalize(student_feat, p=2, dim=-1)
        teacher_feat = F.normalize(teacher_feat, p=2, dim=-1)
        
        comp_loss = self.calc_compression(student_feat, teacher_feat)
        expa_loss = self.calc_expansion(student_feat[:len(teacher_feat)])

        return comp_loss, expa_loss
    
    def calc_compression(self, student_feat_list, teacher_feat_list):
        """
        Compute compression loss between student and teacher features.
        """
        # Convert lists of tensors to a single tensor for vectorized operations
        
        sim = F.cosine_similarity(teacher_feat_list.unsqueeze(1), student_feat_list.unsqueeze(0), dim=-1)
        sim.view(-1, sim.shape[-1])[:: (len(student_feat_list) + 1), :].fill_(0)  # Trick to fill diagonal
        
        n_loss_terms = len(teacher_feat_list)* len(student_feat_list) - min(len(teacher_feat_list), len(student_feat_list))
        # Sum the cosine similarities
        comp_loss = sim.mean(2).sum()/n_loss_terms
        # global_comp_loss = (sim[:, :len(teacher_feat_list)].mean(2).sum()).detach_().div_(len(teacher_feat_list))
        return 1 - comp_loss
    
    def calc_expansion(self, feat_list) -> torch.Tensor:
        """
        Compute expansion loss using Coding Rate estimation.
        """
        cov_list = []
        num_views = len(feat_list)
        _, m, p = feat_list.shape
        
        cov_list = torch.einsum('nbc,nbd->ncd', feat_list, feat_list)

        scalar = p / (m * self.eps)
        I = torch.eye(p, device=cov_list.device)
        loss:torch.Tensor = 0
        for i in range(num_views):
            loss += torch.linalg.cholesky_ex(I + scalar * cov_list[i])[0].diagonal().log().sum()
        loss /= num_views
        # loss *= (p+m)/(p*m) # the balancing factor gamma, you can also use the next line. This is ultimately a heuristic, so feel free to experiment.
        # loss *= ((self.eps * m) ** 0.5 / p)
        loss *= self.eps * np.sqrt(m/(p*np.min([p, m])))
        return -loss