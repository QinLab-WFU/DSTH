import torch
import torch.nn.functional as F
from torch import optim


def distance(x, y, normalize_input=True):
    """
    Args:
        x: embedding tensor
        dist_type: distance metric type
            "cosine" means the negative cosine similarity
            ...
        normalize_input: l2-normalize x or not
    Returns:
        distance matrix, which contains the distance between any two embeddings
    """
    if normalize_input:
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
    if y == None :
        return -(x @ x.T)
    else:
        return -(x @ y.T)
    # elif dist_type == "euclidean":
    #     return torch.cdist(x, x, p=2)
    # elif dist_type == "squared_euclidean":
    #     return ((x.unsqueeze(1) - x.unsqueeze(0)) ** 2).sum(-1)
    # else:
    #     raise NotImplementedError(f"not support: {dist_type}")


def build_optimizer(optim_type, parameters, **kwargs):
    # optimizer_names = ["Adam", "RMSprop", "SGD"]
    # optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    if optim_type == "adam":
        optimizer = optim.Adam(parameters, **kwargs)
    elif optim_type == "rmsprop":
        optimizer = optim.RMSprop(parameters, **kwargs)
    elif optim_type == "adamw":
        optimizer = optim.AdamW(parameters, **kwargs)
    elif optim_type == "sgd":
        optimizer = optim.SGD(parameters, **kwargs)
    else:
        raise NotImplementedError(f"not support: {optim_type}")
    return optimizer
