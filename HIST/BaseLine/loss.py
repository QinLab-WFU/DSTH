from argparse import Namespace

import torch
import torch.nn.functional as F
from torch import nn

from BaseLine.miner import TripletMarginMiner
from BaseLine.utils import distance


class TripletMarginLoss(nn.Module):
    def __init__(self, args: Namespace, **kwargs):
        super().__init__()
        self.margin = args.margin
        self.l2_normalization = args.l2_normalization if "l2_normalization" in args else False
        self.type_of_distance = args.type_of_distance
        self.need_cnt = kwargs.pop("need_cnt", False)
        self.miner = TripletMarginMiner(args)

    def forward(self, embeddings, labels):
        indices_tuple = self.miner(embeddings.detach(), labels)

        anchor_idx, positive_idx, negative_idx = indices_tuple
        # n_triplets = indices_tuple[0].numel()
        n_triplets = len(anchor_idx)
        if len(anchor_idx) == 0:
            # print("no triplets")
            return (0, 0) if self.need_cnt else 0

        mat = distance(embeddings, self.type_of_distance, not self.l2_normalization)
        ap_dists = mat[anchor_idx, positive_idx]
        an_dists = mat[anchor_idx, negative_idx]

        violation = ap_dists - an_dists + self.margin
        losses = F.relu(violation)

        return (torch.mean(losses), n_triplets) if self.need_cnt else torch.mean(losses)
