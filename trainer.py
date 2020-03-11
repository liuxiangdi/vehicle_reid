import torch
import time
import os
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
            self.loader_train = AIC20_dataloader_CCL("train")

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
        validation_model = model (if model is not None) else self.model
        
        # caculate CMC
        features = []



if __name__ == "__main__":
    trainer = CCL_trainer(model_name = "Vgg16", data_name = "AIC20")
    trainer.train()
