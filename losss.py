import torch
import torch.nn.functional as F


def ccl_loss(pos_features, neg_features, margin=0.5):
    # 计算positive 中心点
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