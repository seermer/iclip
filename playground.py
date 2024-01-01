from mmdet.models.losses import FocalLoss
import torch

loss_fn = FocalLoss().to('cuda')
loss_fn2 = torch.nn.CrossEntropyLoss().to('cuda')

x = torch.rand((4, 8)).to('cuda') * 60 - 30
label = torch.randint(0, 8, (4,), dtype=torch.long).to('cuda')

print(loss_fn(x, label))
print(loss_fn2(x, label))



