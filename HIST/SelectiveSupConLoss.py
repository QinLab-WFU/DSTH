
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.cls2emb = nn.Sequential(
            nn.Linear(cls_len,64),
            nn.ReLU(),
            nn.Linear(64,hash_bit)
        )

    def forward(self, x):
        return torch.tanh(self.cls2emb(x))


import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, ):
        super(SupConLoss, self).__init__()
        self.loss = 1
        self.temperature = 0.3
        self.margin = 0.5

    def forward(self, features, prototypes, labels=None,feat2 = None):
        # data-to-data
        anchor_feature = features
        contrast_feature = features if feat2 == None else feat2
        mask = (torch.mm(labels.float(), labels.float().T) > 0).float()
        pos_mask = mask
        neg_mask = 1 - mask
        anchor_dot_contrast = torch.matmul(F.normalize(anchor_feature, dim=1), F.normalize(contrast_feature, dim=1).T)
        all_exp = torch.exp(anchor_dot_contrast / self.temperature)
        pos_exp = pos_mask * all_exp
        neg_exp = neg_mask * all_exp
        # data-to-class
        pos_mask2 = labels
        neg_mask2 = 1 - labels

        # Mask
        for i in range(anchor_dot_contrast.shape[0]):
            pos = pos_mask[i] * anchor_dot_contrast[i]
            neg = neg_mask[i] * anchor_dot_contrast[i]
            pos_min = torch.min(pos)
            neg_max = torch.max(neg)
            # print(pos_min,neg_max)
            for index in range(anchor_dot_contrast.size(1)):
                j = anchor_dot_contrast[i][index]
                if j + self.margin <= pos_min and neg_mask[i][index] == True:  # neg
                    anchor_dot_contrast[i][index] = 0
                if j - self.margin >= neg_max and pos_mask[i][index] == True:  # pos
                    anchor_dot_contrast[i][index] = 0

        anchor_dot_prototypes = torch.matmul(F.normalize(anchor_feature, dim=1), F.normalize(prototypes, dim=1).T)
        all_exp2 = torch.exp(anchor_dot_prototypes / self.temperature)
        pos_exp2 = pos_mask2 * all_exp2
        neg_exp2 = neg_mask2 * all_exp2

        delta = 1
        pos_exp *= torch.exp(-1 - anchor_dot_contrast).detach() ** (delta / 4)
        neg_exp *= torch.exp(-1 + anchor_dot_contrast).detach() ** (delta)
        pos_exp2 *= torch.exp(-1 - anchor_dot_prototypes).detach() ** (delta / 4)
        neg_exp2 *= torch.exp(-1 + anchor_dot_prototypes).detach() ** (delta)

        lambda_pos = pos_mask.sum(1) / pos_mask2.sum(1)
        lambda_neg = neg_mask.sum(1) / neg_mask2.sum(1)
        loss = -torch.log((pos_exp.sum(1) + lambda_pos * pos_exp2.sum(1))
                          / (neg_exp.sum(1) + lambda_neg * neg_exp2.sum(1) + pos_exp.sum(1) + lambda_pos * pos_exp2.sum(
            1)))

        return loss.mean(), anchor_dot_prototypes


loss = SupConLoss()

u = torch.randn((32, 16))
p = torch.randn((10, 16))
# 生成一个形状为(3, 5)的矩阵，每个向量只有一个元素为1，其他元素为0
matrix = torch.zeros(32, 10)
indices = torch.randint(0, 10, (32,))
matrix[torch.arange(32), indices] = 1
print(loss(u, p, matrix))