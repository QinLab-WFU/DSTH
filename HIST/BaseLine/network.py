from argparse import Namespace

import torch
import torch.nn.functional as F
from torch import nn

from _utils import build_default_model, set_attr, get_attr


class L2NormLayer(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)


def build_model(args: Namespace, pretrained):
    net, pos = build_default_model(args.backbone, args.n_bits, pretrained, True)
    if hasattr(args, "l2_normalization") and args.l2_normalization:
        set_attr(net, pos, nn.Sequential(get_attr(net, pos), L2NormLayer()))
    return net.cuda()


if __name__ == "__main__":
    from _utils import init
    from utils import distance
    from loss import TripletMarginLoss

    init("1")

    _args = Namespace(backbone="resnet50", n_bits=32, l2_normalization=True, type_of_triplets="all")
    # for x in ["alexnet", "resnet18", "resnet50", "swin_t", "vit_b_16"]:
    #     _args.backbone = x
    #     net = build_model(_args, False)
    #     print(f"number of {x}'s params: {calc_learnable_params(net)}")

    net1 = build_model(_args, True)
    _args.l2_normalization = False
    net2 = build_model(_args, True)

    _images = torch.rand((5, 3, 224, 224)).cuda()
    _labels = (torch.randn(5, 10) > 0.8).float().cuda()

    with torch.no_grad():
        logits1 = net1(_images)
        logits2 = net1(_images)
        dist1 = distance(logits1, "squared_euclidean", False)
        dist2 = distance(logits1, "cosine")

    print(dist1)
    print(2 + 2 * dist2)

    _args.margin = 0.5
    _args.type_of_distance = "squared_euclidean"
    criteria1 = TripletMarginLoss(_args)
    _args.margin = 0.25
    _args.type_of_distance = "cosine"
    criteria2 = TripletMarginLoss(_args)

    print("norm^2", criteria1(logits1, _labels).item())
    print("cosine", 2 * criteria2(logits1, _labels).item())
