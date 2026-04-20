import torch
from src.layers import PrunableLinear

layer = PrunableLinear(5, 3)

x = torch.randn(2, 5)
output = layer(x)

print("Output shape:", output.shape)