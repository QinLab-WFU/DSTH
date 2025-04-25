import torch
import torch.nn.functional as F



class MultiPosPairLoss():
    """
    Designed for SimSiam.

    paper: `Exploring Simple Siamese Representation Learning <https://arxiv.org/abs/2011.10566>`_
    """

    def __init__(
            self,
            mask_rules=[[True]],
            **kwargs
    ):
        super().__init__(**kwargs)
        self.mask_rules = mask_rules

        self.to_record_list = []
        for i in range(len(mask_rules)):
            self.to_record_list += ["pos_mean_{}".format(i), "neg_mean_{}".format(i), "loss_{}".format(i)]

    def required_metric(self):
        return ["cosine"]

    def generate_mask(self, labels):
        mask = (torch.mm(labels.float(), labels.float().T) > 0).float()
        pos_mask = mask
        neg_mask = 1 - mask


        return pos_mask, neg_mask

    def compute_loss(
            self,
            features,
            labels,
            feat2=None,
    ) -> torch.Tensor:
        anchor_feature = features
        contrast_feature = features if feat2 == None else feat2
        mask = (torch.mm(labels.float(), labels.float().T) > 0).float()
        pos_mask = mask
        neg_mask = 1 - mask
        anchor_dot_contrast = torch.matmul(F.normalize(anchor_feature, dim=1), F.normalize(contrast_feature, dim=1).T)

        pos_exp = pos_mask * anchor_dot_contrast
        neg_exp = neg_mask * anchor_dot_contrast

        total_loss = - pos_exp.sum(1) + neg_exp.sum(1)

        return total_loss.mean()

# loss = MultiPosPairLoss()
# mat = torch.randn((32,32))
# label = torch.randn((32,21))
# value = loss.compute_loss(mat,label)
# print(value)