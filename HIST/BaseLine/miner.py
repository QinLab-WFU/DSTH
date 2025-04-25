from argparse import Namespace

import torch
from torch import nn

from BaseLine.utils import distance


def get_all_triplets_indices(labels):
    # if batch_size = 7
    # A = sames.unsqueeze(2) -> 7 x 7 x 1
    # B = diffs.unsqueeze(1) -> 7 x 1 x 7
    # A * B -> [7 x 7 x 1->repeat->7] * [7 x 1->repeat->7 x 7]
    #           a11 a11..........a11     b11 b12..........b17 : a11=0 -> pass
    #           a12 a12..........a12     b11 b12..........b17 : if a12=1 & b15=1 -> find triplet: 1=anc 2=pos 5=neg
    #           ...
    #           a77...
    # torch.where([7 x 7 x 7]) -> all 1s in the matrix's 3d index: ([x1,...],[y1,...],[z1,...])
    # so xi, yi, zi is index of anchor, positive and negative in the batch
    if len(labels.shape) == 1:
        sames = labels.unsqueeze(1) == labels.unsqueeze(0)
    else:
        sames = (labels @ labels.T > 0).byte()
    diffs = sames ^ 1
    sames.fill_diagonal_(0)
    # NOTE: gen triplets using AP=1 & AN=0 but lack of PN=0, which will not harm the TripletLoss,
    # because another triplet may use P as A.
    return torch.where(sames.unsqueeze(2) * diffs.unsqueeze(1))


class TripletMarginMiner(nn.Module):
    """
    Returns triplets that violate the margin
    Args:
        ...
        args.type_of_triplets: options are "all", "hard", "semi-hard", or "easy".
            "all" means all triplets that violate the margin
            "hard" is a subset of "all", but the negative is closer to the anchor than the positive
            "semi-hard" is a subset of "all", but the negative is further from the anchor than the positive
            "easy" is all triplets that are not in "all"
    """

    def __init__(self, args: Namespace):
        super().__init__()
        self.margin = args.margin
        self.type_of_distance = args.type_of_distance
        self.type_of_triplets = args.type_of_triplets
        self.l2_normalization = args.l2_normalization if "l2_normalization" in args else False

    def forward(self, embeddings, labels):
        anchor_idx, positive_idx, negative_idx = get_all_triplets_indices(labels)
        mat = distance(embeddings.detach(), self.type_of_distance, not self.l2_normalization)
        ap_dist = mat[anchor_idx, positive_idx]
        an_dist = mat[anchor_idx, negative_idx]
        triplet_margin = an_dist - ap_dist

        if self.type_of_triplets == "easy":
            threshold_condition = triplet_margin > self.margin
        else:
            threshold_condition = triplet_margin <= self.margin
            if self.type_of_triplets == "hard":
                threshold_condition &= triplet_margin <= 0
            elif self.type_of_triplets == "semi-hard":
                threshold_condition &= triplet_margin > 0
            else:
                pass  # here is "all"

        return (
            anchor_idx[threshold_condition],
            positive_idx[threshold_condition],
            negative_idx[threshold_condition],
        )
