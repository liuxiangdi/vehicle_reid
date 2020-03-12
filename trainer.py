import torch
import time
import os
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from model import Vgg16Net
from dataloader import AIC20_dataloader_CCL
from losss import ccl_loss

device = torch.device("cuda:1")
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
        self.model.to(device)
        
        if data_name == "AIC20":
            self.loader_train = AIC20_dataloader_CCL("train")
            self.loader_test = AIC20_dataloader_CCL("test")

    def train(self):
        lr = 0.001
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.8)
        avg_loss = 0
        for index in range(20000):
            if index % 10000 == 0:
                lr /= 2
                optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.8)
            
            pos, neg = self.loader_train.get_batch()
            pos = torch.stack(pos).to(device)
            neg = torch.stack(neg).to(device)

            pos_features = self.model(pos)
            neg_features = self.model(neg)

            optimizer.zero_grad()
            loss = ccl_loss(pos_features, neg_features)
            
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            if index % 2000 == 0 and index != 0:
                path = os.path.join(model_path, "{}_{}_{}_{}.pt".format(self.model_name, self.data_name, index, avg_loss))
                torch.save(self.model.state_dict(), path)
            if index % 100 == 0 and index != 0:
                print('batch {}  avgloss {}'.format(index, avg_loss/100))
                avg_loss = 0

    def validation(self, model=None):
        validation_model = model if model is not None else self.model

        # caculate CMC
        # 提取所有features
        t = time.time()
        features = []
        for batch, inputs in enumerate(self.loader_test):
            inputs = inputs.to(device)
            outputs = self.model(inputs).cpu().numpy()
            for index in range(len(outputs)):
                feature = outputs[index]
                feature /= np.linalg.norm(feature, 2)
                features.append([feature, batch])
        print("extract feature time  {}".format(time.time() - t))

        # 计算所有features的dis
        t = time.time()
        diss = np.ones([len(features), len(features)], dtype=np.float)
        for i in range(len(features)):
            for j in range(i+1, features):
                dis = np.linalg.norm(features[i], features[j])
                diss[i][j] = dis
                diss[j][i] = dis
        print("cal dis time  {}".format(time.time() - t))


if __name__ == "__main__":
    trainer = CCL_trainer(model_name = "Vgg16", data_name = "AIC20")
    trainer.train()
