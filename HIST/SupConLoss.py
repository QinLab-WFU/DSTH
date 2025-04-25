import torch
import torch.nn as nn
import torch.nn.functional as F


class Emb2Class(nn.Module):
    def __init__(self, cls_len, hash_bit, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """_summary_

        Returns:
            _type_: _description_
        """
        self.emb = nn.Linear(hash_bit, 512)
        self.lrelu = nn.LeakyReLU()
        self.cls = nn.Linear(512, cls_len)

    def forward(self, x):
        return self.cls(self.lrelu(self.emb(x)))


class Class2Emb(nn.Module):
    def __init__(self, cls_len, hash_bit, *args, **kwargs) -> None:
        """_summary_

        Args:
            cls_len (int): class
            hash_bit (bool): output_dim of the hashcode

        Example:
            input_tensor = torch.eye(cls_len) --> shape : [cls_len , cls_len]
            prototype = model(input_tensor) --> shape : [cls_len , hash_bit]
        """
        super().__init__(*args, **kwargs)
        self.cls2emb = nn.Linear(cls_len, hash_bit)

    def forward(self, x):
        return torch.tanh(self.cls2emb(x))


def smooth_CE(logits, label, peak=0.9):
    """_summary_

    Args:
        logits (torch.Tensor): prediction --> shape : [Batch, num_cls]
        label (torch.Tensor): one_hot label --> shape : [Batch, num_cls]
        peak (float): used for smooth the input label --> default : 0.9

    Returns:
        loss: loss with gradient
    """
    batch, num_cls = logits.size()

    label_logits = label
    smooth_label = torch.ones(logits.size()) * (1 - peak) / (num_cls - 1)
    smooth_label[label_logits == 1] = peak

    logits = F.log_softmax(logits, -1)
    ce = torch.mul(logits, smooth_label.to(logits.device))
    loss = torch.mean(-torch.sum(ce, -1))  # batch average

    return loss



class SupConLoss(nn.Module):
    def __init__(self, loss=1, temperature=0.3, data_class=10):
        super(SupConLoss, self).__init__()
        self.thresh = 0.5
        self.loss = loss
        self.temperature = temperature
        self.data_class = data_class

    def forward(self, features, prototypes, labels, epoch):
        # data-to-data
        anchor_feature = features
        contrast_feature = features
        mask = (torch.mm(labels.float(), labels.float().T) > 0).float()
        pos_mask = mask  # 正样本索引
        neg_mask = 1 - mask  # 负样本索引
        anchor_dot_contrast = torch.matmul(F.normalize(anchor_feature, dim=1), F.normalize(contrast_feature, dim=1).T)
        all_exp = torch.exp(anchor_dot_contrast / self.temperature)  # 缩放 all > 0
        pos_exp = pos_mask * all_exp  # 正负样本的特征值矩阵
        neg_exp = neg_mask * all_exp
        # data-to-class
        pos_mask2 = labels
        neg_mask2 = 1 - labels
        anchor_dot_prototypes = torch.matmul(F.normalize(anchor_feature, dim=1), F.normalize(prototypes, dim=1).T)
        all_exp2 = torch.exp(anchor_dot_prototypes / self.temperature)
        pos_exp2 = pos_mask2 * all_exp2
        neg_exp2 = neg_mask2 * all_exp2

        # self_paced:
        if epoch <= int(100 / 3):  # 100 == epoches
            delta = epoch / int(100 / 3)
        else:
            delta = 1
        pos_exp *= torch.exp(-self.thresh - anchor_dot_contrast).detach() ** (delta / 4)
        neg_exp *= torch.exp(-self.thresh + anchor_dot_contrast).detach() ** (delta)
        pos_exp2 *= torch.exp(-self.thresh - anchor_dot_prototypes).detach() ** (delta / 4)
        neg_exp2 *= torch.exp(-self.thresh + anchor_dot_prototypes).detach() ** (delta)

        lambda_pos = pos_mask.sum(1) / pos_mask2.sum(1)
        lambda_neg = neg_mask.sum(1) / neg_mask2.sum(1)
        loss = -torch.log((pos_exp.sum(1) + lambda_pos * pos_exp2.sum(1))
                          / (neg_exp.sum(1) + lambda_neg * neg_exp2.sum(1)
                             + pos_exp.sum(1) + lambda_pos * pos_exp2.sum(1)))
        return loss.mean()

# if self.loss == 'p2p':
#     loss = -torch.log(pos_exp.sum(1)/(neg_exp.sum(1) + pos_exp.sum(1)))
#     return loss.mean()
# if self.loss == 'p2c':
#     loss = -torch.log(pos_exp2.sum(1)/(neg_exp2.sum(1) + pos_exp2.sum(1)))
#     return loss.mean()
# if self.loss == 'RCH':
#     # balance two kinds of pairs
#     if opt.weighting:
#         lambda_pos = pos_mask.sum(1)/pos_mask2.sum(1)
#         lambda_neg = neg_mask.sum(1)/neg_mask2.sum(1)
#         loss = -torch.log((pos_exp.sum(1) + lambda_pos*pos_exp2.sum(1))
#                           / (neg_exp.sum(1) + lambda_neg*neg_exp2.sum(1) + pos_exp.sum(1) + lambda_pos*pos_exp2.sum(1)))
#     else:
#         loss = -torch.log((pos_exp.sum(1) + pos_exp2.sum(1))
#                           / (neg_exp.sum(1) + neg_exp2.sum(1) + pos_exp.sum(1) + pos_exp2.sum(1)))




