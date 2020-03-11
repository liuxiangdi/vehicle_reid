import torch
import torch.optim as optim
import torch.nn.functional as F
from model import Vgg16Net
from dataloader import CClDataLoader

device = torch.device("cuda:1")

class CCL_trainer():
    def __init__(self):
        super().__init__()
        self.model = Vgg16Net()
        self.model.to(device)
    """
    def ccl_loss(self, pos_features, neg_features, margin=0.5):
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
        return loss
    """

    def ccl_loss(self, positiveimageset, negetiveimageset, margin):
        posbatchsize = len(positiveimageset)
        negbatchsize = len(negetiveimageset)
        posimgbatch = 0
        negimgbatch = 0
        for i in range(posbatchsize):
            if i == 0:
                posimgbatch = positiveimageset[0]
            else:
                posimgbatch = torch.cat([posimgbatch, positiveimageset[i]])
        for i in range(negbatchsize):
            if i == 0:
                negimgbatch = negetiveimageset[0]
            else:
                negimgbatch = torch.cat([negimgbatch, negetiveimageset[i]])

        posfeatures = self.model(posimgbatch)
        negfeatures = self.model(negimgbatch)

        centerfeature = 0
        for i in range(len(posfeatures)):
            if i == 0:
                centerfeature = posfeatures[0]
            else:
                centerfeature = centerfeature + posfeatures[i]
        centerfeature = torch.div(centerfeature, len(posfeatures))

        #归一化
        centerfeature = torch.div(centerfeature, torch.norm(centerfeature, 2))
        centerfeature = torch.unsqueeze(centerfeature, 0)
        #print(centerfeature.shape)

        tempnegfeature = 0
        harddistance = float('Inf')
        hardnegefeature = Variable(torch.zeros(1, 128).cuda())
        for i in range(len(negfeatures)):
            tempnegfeature = negfeatures[i]
            #归一化
            tempnegfeature = torch.div(tempnegfeature, torch.norm(tempnegfeature, 2))
            tempnegfeature = torch.unsqueeze(tempnegfeature, 0)
            #print(tempnegfeature.shape)
            dis = F.pairwise_distance(centerfeature, tempnegfeature, p=2).cpu().data.numpy()[0][0]
            if dis < harddistance:
                harddistance = dis
                hardnegefeature = tempnegfeature

        totalloss = 0
        for i in range(len(posfeatures)):
            temposfeature = posfeatures[i]
            #归一化
            temposfeature = torch.div(temposfeature, torch.norm(temposfeature, 2))
            temposfeature = torch.unsqueeze(temposfeature, 0)
            loss = F.pairwise_distance(temposfeature, centerfeature) + margin - F.pairwise_distance(centerfeature, hardnegefeature)
            loss = torch.clamp(loss, min=0)
            totalloss += loss
        return totalloss

    def train(self):
        loader = CClDataLoader("train")
        
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.8)
        avg_loss = 0
        for index in range(10000):
            pos, neg = loader.get_batch()
            pos = torch.stack(pos).to(device)
            neg = torch.stack(neg).to(device)

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
