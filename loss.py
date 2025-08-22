import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-6)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        with torch.amp.autocast('cuda', enabled=False):
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
            I = self.pairwise_NNs_inner(student_output)  # noqa: E741
            distances = self.pdist(student_output, student_output[I])  # BxD, BxD -> B
            distances = torch.clamp(distances, min=eps)
            loss = -torch.log(distances + eps).mean()
        return loss


class MCRLoss(nn.Module):
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
        loss:torch.Tensor = 0

        I = torch.eye(p, device=cov_list.device, dtype=cov_list.dtype)
        for i in range(num_views):
            mat = (I + scalar * cov_list[i]).to(torch.float32)
            loss_term = torch.linalg.cholesky_ex(mat)[0].diagonal().log().sum()
            loss += loss_term.to(dtype=cov_list.dtype)  # back to original dtype

        loss /= num_views
        # loss *= (p+m)/(p*m) # the balancing factor gamma, you can also use the next line. This is ultimately a heuristic, so feel free to experiment.
        # loss *= ((self.eps * m) ** 0.5 / p)
        loss *= self.eps * np.sqrt(m/(p*np.min([p, m])))
        return -loss