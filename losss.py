import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from itertools import combinations

def ccl_loss(pos_features, neg_features, margin=1.0):
    
    # 计算positive 中心点
    """
    pos_center = None
    for i in range(len(pos_features)):
        _feature = neg_features[i]
        _feature = torch.div(_feature, torch.norm(_feature, 2))
        if i == 0:
            pos_center = _feature
            continue
        pos_center += _feature
    pos_center = torch.div(pos_center, torch.norm(pos_center, 2))
    pos_center = torch.unsqueeze(pos_center, 0)
    """
    pos_center = pos_features.mean(0)
    pos_center = torch.div(pos_center, torch.norm(pos_center, 2))
    pos_center = torch.unsqueeze(pos_center, 0)

    # 计算Hard negative feature
    hard_neg = None
    hard_dis = float("inf")
    for i in range(len(neg_features)):
        _feature = neg_features[i]
        _feature = torch.div(_feature, torch.norm(_feature, 2))
        _feature = torch.unsqueeze(_feature, 0)
        dis = F.pairwise_distance(pos_center, _feature, p=2).cpu().data.numpy()[0]
        if dis < hard_dis:
            hard_dis = dis
            hard_neg = _feature
    
    total_loss = 0
    for i in range(len(pos_features)):
        pos_feature = pos_features[i]
        pos_feature = torch.div(pos_feature, torch.norm(pos_feature, 2))
        pos_feature = torch.unsqueeze(pos_feature, 0)
        loss = F.pairwise_distance(pos_feature, pos_center) + margin - F.pairwise_distance(pos_center, hard_neg)
        loss = torch.clamp(loss, min=0)

        total_loss += loss
    return total_loss


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix



class BatchHardTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
    
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.5):
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        # image num 为5
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt() # for numerical stability
        # dis 为距离矩阵，nxn维

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        # mask是一个表示是否同ID的矩阵, 如果ID为 [1,2,1,3,4], mask 为
        # [[ True, False,  True, False, False],
        # [False,  True, False, False, False],
        # [ True, False,  True, False, False],
        # [False, False, False,  True, False],
        # [False, False, False, False,  True]]

        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        # 对于每一张图片,选择farest ap和 farest an,保存距离,如果没有ap或者没有an, 则为0
        # 应一次输入n个class, 每个class m张
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)

if __name__ == "__main__":
    inputs = torch.tensor([[1,2,3.0], [1,2,4], [1,2,4], [2,3,4], [5,6,7]])
    labels = torch.tensor([1,2,1,3,4])

    model = BatchHardTripletLoss()
    dis = model(inputs, labels)
