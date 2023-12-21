import torch

labels = torch.stack([torch.arange(9) for _ in range(4)])
print(labels)
labels = labels + torch.arange(0, labels.size(0) * labels.size(1), labels.size(1)).unsqueeze(1)
print(labels)
