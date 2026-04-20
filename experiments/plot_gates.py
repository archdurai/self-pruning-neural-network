import torch
import matplotlib.pyplot as plt
from src.train import train_model

# Train model (use best lambda based on your experiments)
model, acc, sparsity = train_model(lambda_val=0.001, epochs=3)

all_gates = []

# Collect all gate values
for module in model.modules():
    if hasattr(module, 'gate_scores'):
        gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy()
        all_gates.extend(gates.flatten())

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(all_gates, bins=50)

plt.title("Gate Value Distribution")
plt.xlabel("Gate Value")
plt.ylabel("Frequency")

# Save the plot (IMPORTANT)
plt.savefig("experiments/gate_distribution.png")

# Show plot
plt.show()