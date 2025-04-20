import torch
import torch.nn.functional as F
from torchmetrics.regression import ConcordanceCorrCoef


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def vicreg_loss(embedding, reduction='mean'):
    # subtracting the mean along the batch dimension
    embedding = embedding - embedding.mean(dim=0, keepdim=True)

    # getting the maxpool along the batch dimension
    embedding = embedding.max(dim=1, keepdim=False)[0]
    
    batch_size, num_features = embedding.shape

    std_x = torch.sqrt(embedding.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std_x))

    cov_x = (embedding.T @ embedding) / (batch_size - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum().div(num_features) 

    if reduction == "mean":
        std_loss = std_loss.mean()
        cov_loss = cov_loss.mean()
    elif reduction == "sum":
        std_loss = std_loss.sum()
        cov_loss = cov_loss.sum()

    return std_loss, cov_loss


def embedding_cross_entropy_loss(input, target, mask=None, reduction='mean'):
    loss = torch.sum(F.softmax(target, dim=-1) * F.log_softmax(input, dim=-1), dim=-1)
    if mask is not None:
        loss = loss[mask]
    if reduction == "mean":
        loss = -loss.mean()
    elif reduction == "sum":
        loss = -loss.sum()
    else:
        loss = -loss
    return loss
    

def masked_mse_loss(input, target, reduction='mean', mask=None):
    out = (input - target)**2
    # do not consider elements to 0 from the input
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
        out = out[mask]
    if reduction == "mean":
        return out.mean()
    elif reduction == "sum":
        return out.sum()
    else:
        return out
    
def masked_min_max_loss(input, target, reduction='mean', patch_size=100):
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
    out = out[max_target != 0]

    if reduction == "mean":
        return out.mean() / tokens_num
    elif reduction == "sum":
        return out.sum() / tokens_num
    else:
        return out / tokens_num
    
def gradient_loss(input, target, reduction='mean', p=2):
    input_grad = input[:, 1:] - input[:, :-1]
    target_grad = target[:, 1:] - target[:, :-1]
    out = torch.pow(input_grad - target_grad, p)
    # do not consider elements that was at 0 in the input
    # using [:, :-1] because preserve the order of the elements
    out = out[target[:, :-1] != 0]
    if reduction == "mean":
        return out.mean()
    elif reduction == "sum":
        return out.sum()
    else:
        return out
    


