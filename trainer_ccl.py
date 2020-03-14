import torch
import sys
import time
import os
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from model import Vgg16Net, MobileNet, ResNet50, Vgg11Net, ResNet34
from dataloader import AIC20_dataloader_CCL, PersonDataloader
from losss import ccl_loss

device = torch.device("cuda:0")
date = time.strftime("%m-%d", time.localtime())
model_path = "/home/lxd/checkpoints/" + date
if not os.path.exists(model_path):
    os.makedirs(model_path)

class CCL_trainer():
    def __init__(self, model_name, data_name):
        super().__init__()
        self.model_name = model_name
        self.data_name = data_name

        if model_name == "Vgg16":
            self.model = Vgg16Net()
        elif model_name == "MobileNet":
            self.model = MobileNet()
        elif model_name == "ResNet50":
            self.model = ResNet50()
        elif model_name == "ResNet34":
            self.model = ResNet34()
        elif model_name == "Vgg11":
            self.model = Vgg11Net()
        else:
            print("!!!  Model Wrong  !!!")

        print("{} init".format(model_name))
        self.model.to(device)
        
        if data_name == "AIC20":
            self.loader_train = AIC20_dataloader_CCL("train")
        elif data_name == "Market-1501":
            self.loader_train = PersonDataloader("train")

    def train(self):
        lr = 0.001
        #optimizer = optim.Adam(self.model.parameters(), lr=lr)
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.8)
        avg_loss = 0
        for index in range(50000):
            if index % 10000 == 0:
                lr /= 10
                optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.8)
            
            pos, neg = self.loader_train.get_batch()
            pos = pos.to(device)
            neg = neg.to(device)

            pos_features = self.model(pos)
            neg_features = self.model(neg)

            optimizer.zero_grad()
            loss = ccl_loss(pos_features, neg_features)
            
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            if index % 2000 == 0 and index != 0:
                path = os.path.join(model_path, "{}_{}_{}.pt".format(self.model_name, self.data_name, index))
                torch.save(self.model.state_dict(), path)
            if index % 100 == 0 and index != 0:
                print('batch {}  avgloss {}'.format(index, avg_loss/100))
                avg_loss = 0

    def validation(self, model=None):
        # 对于每个ID车辆只取三张图
        print("--------- validation -----------")
        #loader_test = AIC20_dataloader_CCL("test")
        loader_test = PersonDataloader("test")
        validation_model = model if model is not None else self.model
        # caculate CMC
        # 提取所有features
        t = time.time()
        features = []
        for car_id in range(loader_test.get_num()):
            #if car_id % 50 == 0 and car_id != 0:
            #    break
            inputs = loader_test.get_batch().to(device)
            outputs = validation_model(inputs).cpu().detach().numpy()
            for index in range(len(outputs)):
                feature = outputs[index]
                feature = feature / np.linalg.norm(feature, 2)
                features.append([feature, car_id])
        print("extract feature time  {}".format(time.time() - t))

        # 计算所有features的dis
        t = time.time()
        diss = np.full([len(features), len(features)], float("Inf"), dtype=np.float)
        for i in range(len(features)):
            if i % 100 == 0:
                print("cal_dis {} / {}".format(i, len(features)))
            for j in range(i+1, len(features)):
                dis = float(np.linalg.norm(features[i][0]-features[j][0], 2))
                diss[i][j] = dis
                diss[j][i] = dis
        print("cal dis time  {}".format(time.time() - t))
        
        t = time.time()
        cms_lis = []
        for index, dis in enumerate(diss):
            anchor_ID = features[index][1]
            dis = [[j, i] for i,j in enumerate(list(dis))]
            dis.sort()
            #if index % 100 == 0:
            #    print("{} / {}".format(index, len(features)))
            for ranki in range(10):
                match_ID = features[dis[ranki][1]][1]
                if match_ID == anchor_ID:
                    cms_lis.append(ranki)
                    break
        print("cms_lis time {}".format(time.time() - t))
        ans = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in cms_lis:
            ans[i] += 1/len(features)
        for i in range(9):
            ans[i+1] += ans[i]
        print(ans)


dataloader = PersonDataloader("train")
anchor, pos, neg = dataloader.get_triplet_batch()
anchor = anchor.to(device)
pos = pos.to(device)
neg = neg.to(device)

model = Vgg16Net()
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
avg_loss = 0
for index in range(101):
    pos_features = model(pos)
    anchor_features = model(anchor)
    neg_features = model(neg)

    optimizer.zero_grad()
    loss = F.triplet_margin_loss(anchor_features, pos_features, neg_features)
    
    loss.backward()
    optimizer.step()

    avg_loss += loss.item()
    if index % 100 == 0 and index != 0:
        print('batch {}  avgloss {}'.format(index, avg_loss/100))
        avg_loss = 0

batch_out1 = model(anchor)
batch_out2 = model(pos)
batch_out3 = model(neg)
#print(batch_out1)
#print(batch_out2)
print("same")
for i in range(4):
    print((batch_out1[i] - batch_out2[i]).pow(2).sum())
print("different")
for i in range(4):
    print((batch_out1[i] - batch_out3[i]).pow(2).sum())

if __name__ == "___main__":
    model = sys.argv[1]
    mode = sys.argv[2]
    batch = sys.argv[3]
    date = "03-14"
    
    if mode == "train":
        trainer = None
        if model == "mobile":
            trainer = CCL_trainer(model_name = "MobileNet", data_name = "Market-1501")
        elif model == "vgg16":
            trainer = CCL_trainer(model_name = "Vgg16", data_name = "Market-1501")
        elif model == "res34":
            trainer = CCL_trainer(model_name = "ResNet34", data_name = "Market-1501")
        trainer.train()
    elif mode == "test":
        if model == "mobile":
            model = MobileNet()
            model.load_state_dict(torch.load("/home/lxd/checkpoints/{}/MobileNet_Market-1501_{}.pt".format(date, batch)))
            model.to(device)
            model.eval()
            trainer = CCL_trainer(model_name = "MobileNet", data_name = "Market-1501")
            trainer.validation(model)
        elif model == "vgg16":
            model = Vgg16Net()
            model.load_state_dict(torch.load("/home/lxd/checkpoints/{}/Vgg16_Market-1501_{}.pt".format(date, batch)))
            model.to(device)
            model.eval()
            trainer = CCL_trainer(model_name = "Vgg16", data_name = "Market-1501")
            trainer.validation(model)
        elif model == "res34":
            model = ResNet34()
            model.load_state_dict(torch.load("/home/lxd/checkpoints/{}/ResNet34_Market-1501_{}.pt".format(date, batch)))
            model.to(device)
            model.eval()
            trainer = CCL_trainer(model_name = "ResNet34", data_name = "Market-1501")
            trainer.validation(model)
        

    """
    #trainer = CCL_trainer(model_name = "MobileNet", data_name = "Market-1501")
    #trainer.train()
    model = MobileNet()
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load("/home/lxd/checkpoints/03-13/MobileNet_Market-1501_4000.pt"))
    trainer = CCL_trainer(model_name = "MobileNet", data_name = "Market-1501")
    trainer.validation(model)
    """

    """
    trainer = CCL_trainer(model_name = "ResNet34", data_name = "Market-1501")
    trainer.train()
    model = ResNet34()
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load("/home/lxd/checkpoints/03-13/ResNet34_Market-1501_2000.pt"))
    trainer = CCL_trainer(model_name = "ResNet34", data_name = "Market-1501")
    trainer.validation(model)
    """
    
    """ VGG11
    #trainer = CCL_trainer(model_name = "Vgg16", data_name = "Market-1501")
    #trainer.train()
    
    model = Vgg16Net()
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load("/home/lxd/checkpoints/03-13/Vgg16_Market-1501_4000.pt"))
    trainer = CCL_trainer(model_name = "Vgg16", data_name = "Market-1501")
    trainer.validation(model)
    """
