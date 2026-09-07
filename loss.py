import torch
import torch.nn.functional as F

def graph_recon_loss(adj_recon_logits, adj_orig, epoch):
    """
    adj_recon_logits: [N, N] logits
    adj_orig:         [N, N] {0,1} dense
    epoch:            int, used to change negative sampling each epoch
    """
    device = adj_recon_logits.device
    N = adj_orig.size(0)

    # Masks
    pos_mask = (adj_orig == 1)
    neg_mask = (adj_orig == 0)

    # Positive indices
    pos_idx = pos_mask.nonzero(as_tuple=False)  # [P, 2]
    P = pos_idx.size(0)
    if P == 0:
        return torch.zeros((), device=device)

    # Negative indices
    neg_idx_all = neg_mask.nonzero(as_tuple=False)  # [Z, 2]
    Z = neg_idx_all.size(0)
    if Z == 0:
        # no zeros to sample from
        return torch.zeros((), device=device)

    # Sample same number of negatives; change every epoch
    g = torch.Generator(device=device)
    g.manual_seed(int(epoch))  # ensures different sample across epochs (and reproducible)

    if Z >= P:
        sel_pos = torch.randperm(P, generator=g, device=device)[:round(0.8*P)]
        pos_idx = pos_idx[sel_pos]
        
        sel = torch.randperm(Z, generator=g, device=device)[:round(0.8*P)]
        neg_idx = neg_idx_all[sel]
   
    #print('neg_idx:', neg_idx)
    # Gather logits
    pos_logits = adj_recon_logits[pos_idx[:, 0], pos_idx[:, 1]]
    neg_logits = adj_recon_logits[neg_idx[:, 0], neg_idx[:, 1]]

    logits = torch.cat([pos_logits, neg_logits], dim=0)
    targets = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)

    # BCE on sampled edges only
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss