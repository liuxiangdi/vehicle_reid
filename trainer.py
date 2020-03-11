import torch
import torch.optim as optim
import torch.nn.functional as F
from model import Vgg16Net
from dataloader import CClDataLoader

class CCL_trainer():
    def __init__(self):
        super().__init__()
        self.model = Vgg16Net()
    
    def ccl_loss(self, pos_features, neg_features, margin=0.5):
        # 计算positive 中心点
        pos_center = pos_features.mean(0)
        print("pos_center :", pos_center.shape)
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
    
        print('dis')
        print(dis)
        
        total_loss = 0
        for i in range(len(pos_features)):
            pos_feature = pos_features[i]
            pos_feature = torch.div(pos_feature, torch.norm(pos_feature, 2))
            pos_feature = torch.unsqueeze(pos_feature, 0)
            loss = F.pairwise_distance(pos_feature, pos_center) + margin - F.pairwise_distance(pos_center, hard_neg)
            loss = torch.clamp(loss, min=0)

            total_loss += loss
        print("total loss", loss)
        return loss
    

    def train(self):
        loader = CClDataLoader("train")
        
        optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.8)
        avg_loss = 0
        for index in range(10000):
            print(index)
            pos, neg = loader.get_batch()
            pos = torch.stack(pos)
            neg = torch.stack(neg)

            pos_features = self.model(pos)
            neg_features = self.model(neg)

            optimizer.zero_grad()
            loss = self.ccl_loss(pos_features, neg_features)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            if index %50 == 0 and index != 0:
                print('avgloss {}'.format(avg_loss/50))
                avg_loss = 0

if __name__ == "__main__":
    trainer = CCL_trainer()
    trainer.train()