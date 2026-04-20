# Results: Self-Pruning Neural Network

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|-------------|-------------|
| 0.01   | 47.34       | 4.40        |
| 0.1    | 45.64       | 24.84       |
| 0.5    | 45.89       | 52.90       |

## Observations

- Increasing lambda significantly increases sparsity.
- Higher lambda forces the model to prune more connections.
- Accuracy remains relatively stable even with high sparsity.
- The model successfully retains important connections while removing redundant ones.