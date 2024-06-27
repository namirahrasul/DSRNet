from .net_utils import *
from .util import *


def load_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path)
    if 'weight' in checkpoint:
        checkpoint = checkpoint['weight']
    ckpt = {}
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            ckpt[k[7:]] = v
        else:
            ckpt[k] = v
    model.load_state_dict(ckpt) #strict=False by nami
