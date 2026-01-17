import torch
from torchvision.datasets import CIFAR10

# Change this to where CODA expects the data folder
DATA_ROOT = "./data_simple"

# Load CIFAR-10 test set
dataset = CIFAR10(
    root=DATA_ROOT,
    train=False,
    download=True
)

# Extract labels
labels = torch.tensor(dataset.targets, dtype=torch.long)

print("Label shape:", labels.shape)  # Should print: torch.Size([10000])
print("First 10 labels:", labels[:10])

# Save in the format CODA expects
torch.save(labels, f"{DATA_ROOT}/cifar10_5592_label.pt")

print("Saved to:", f"{DATA_ROOT}/cifar10_5592_label.pt")
