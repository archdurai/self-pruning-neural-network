import torch

# Stronger sparsity loss (no normalization → λ has real impact)
def compute_sparsity_loss(model):
    loss = 0.0

    for module in model.modules():
        if hasattr(module, 'gate_scores'):
            gates = torch.sigmoid(module.gate_scores)
            loss += gates.mean()   # keep mean (stable), but no division by layers

    return loss


# Sparsity metric
def calculate_sparsity(model, threshold=0.1):
    total = 0
    pruned = 0

    for module in model.modules():
        if hasattr(module, 'gate_scores'):
            gates = torch.sigmoid(module.gate_scores)

            total += gates.numel()
            pruned += (gates < threshold).sum().item()

    return 100 * pruned / total