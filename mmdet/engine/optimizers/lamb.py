from mmdet.registry import OPTIMIZERS
import torch_optimizer as optim

Lamb = OPTIMIZERS.register_module()(optim.Lamb)
