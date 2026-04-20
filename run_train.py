from src.train import train_model

# Stronger lambda values
lambdas = [0.01, 0.1, 0.5]

results = []

for lam in lambdas:
    print(f"\nRunning for lambda = {lam}")
    model, acc, sparsity = train_model(lambda_val=lam, epochs=5)

    results.append((lam, acc, sparsity))

print("\n===== FINAL COMPARISON =====")
for lam, acc, sparsity in results:
    print(f"Lambda: {lam} | Accuracy: {acc:.2f}% | Sparsity: {sparsity:.2f}%")