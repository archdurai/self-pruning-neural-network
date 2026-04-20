import torch
from src.model import PrunableNet
from src.utils import compute_sparsity_loss   # ← NEW LINE

model = PrunableNet()

x = torch.randn(4, 3, 32, 32)
out = model(x)

# NEW PART
loss = compute_sparsity_loss(model)

print("Output shape:", out.shape)
print("Sparsity loss:", loss.item())