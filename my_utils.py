import torch
from torch import nn
import models
from collections import OrderedDict
import numpy as np

from sklearn.model_selection import KFold

def get_model(model_name, state_dict=None, device=None, num_classes=10, k=None, **kwargs):
    if k is None:
        if "VGG" in model_name:
            k = 64
        elif "PreResNet" in model_name:
            k = 1
        else:
            k = 1
    if isinstance(state_dict, nn.Module):
        state_dict = state_dict.state_dict()
    architecture = getattr(models, model_name)()
    architecture.kwargs["k"] = k
    architecture.kwargs.update(kwargs)
    print(architecture.kwargs)
    if state_dict is not None:
        num_classes = list(state_dict.items())[-1][1].shape[0]
    base = architecture.base(num_classes=num_classes, **architecture.kwargs).to(device)
    if state_dict is not None:
         base.load_state_dict(state_dict)
    return base

def sd2tensor(sd):
    res = None
    for key in sd:
        if res is None:
            res = sd[key].new_zeros(0)
        res = torch.cat((res, sd[key].reshape(-1).float()))
    return res

def get_abc(model, last_count=2):
    a, b, c = 0, 0, 0
    first = True
    num_p = len(list(model.parameters()))
    for i, p in enumerate(model.parameters()):
        # convs
        last = (i >= num_p - last_count)
        if len(p.shape) == 4 and not (first or last):
            a += np.prod(p.shape)
        elif len(p.shape) == 4 and (first or last):
            b += np.prod(p.shape)
        # bias or bn
        elif len(p.shape) == 1 and not (first or last):
            b += np.prod(p.shape)
        elif len(p.shape) == 1 and (first or last):
            c += np.prod(p.shape)
        # linear
        elif len(p.shape) == 2 and not (first or last):
            a += np.prod(p.shape)
        elif len(p.shape) == 2 and (first or last):
            b += np.prod(p.shape)
        first = False
    return a, b, c

def get_size(model_name, num_classes, last_count=2, k=1, device="cpu", **kwargs):
    model = get_model(model_name, None, device, num_classes, k=k, **kwargs)
    a, b, c = get_abc(model, last_count)
    return a+b+c

def compute_k(model_name, N, num_classes, last_count=2, device="cpu"):
    model = get_model(model_name, None, device, num_classes, k=1)
    a, b, c = get_abc(model, last_count)
    return int(np.round(((-b + np.sqrt(4*a*N+(b**2-4*a*c)))/2/a)))

def load(cpt, use_cuda_if_av=True):
    m = torch.load(cpt, map_location="cuda" if torch.cuda.is_available() and use_cuda_if_av else 'cpu')
    if "state_dict" in m:
        m = m["state_dict"]
    return m

def split_loader(loader, num_splits=1):
    dataset = loader.dataset.data
    target = np.array(loader.dataset.targets)
    kfolds = KFold(n_splits=num_splits)
    datasets = []
    targets = []
    idxs = []
    for tr_idx, te_idx in kfolds.split(np.arange(dataset.shape[0])):
        datasets.append(dataset[te_idx])
        targets.append(target[te_idx].tolist())
        idxs.append(te_idx)
    return idxs#datasets, targets