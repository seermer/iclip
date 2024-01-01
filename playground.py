import torch

x = torch.rand((3, 8), dtype=torch.float32) * 120 - 60
labels = torch.randint(0, 7, (3, 1))
labels_onehot = torch.zeros_like(x).scatter_(1, labels, 1)
print(labels)
print(labels_onehot)

ce = torch.nn.CrossEntropyLoss()
bce = torch.nn.BCEWithLogitsLoss()

print(ce(x, labels.squeeze()))
print(bce(x, labels_onehot))
