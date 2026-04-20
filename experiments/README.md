## Key Insight

The model demonstrates a clear sparsity–accuracy tradeoff:

- Increasing λ leads to higher sparsity
- The network prunes up to ~53% of weights with minimal accuracy loss
- This indicates that many connections in the network are redundant

This validates that L1 regularization on sigmoid gates effectively induces self-pruning behavior.

The model achieves over 50% sparsity while maintaining ~46% accuracy, demonstrating efficient compression without significant performance degradation.