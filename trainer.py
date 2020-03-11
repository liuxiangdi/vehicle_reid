import torch
import os
import torch.optim as optim
import torch.nn.functional as F
from model import Vgg16Net
from dataloader import AIC20_dataloader_CCL
from losss import ccl_loss

device = torch.device("cuda:1")
model_path = "/home/lxd/checkpoints"

class CCL_trainer():
    def __init__(self):
        super().__init__()
        self.model = Vgg16Net()
        self.model.to(device)
    
    def train(self):
        loader = AIC20_dataloader_CCL("train")
        
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.8)
        avg_loss = 0
        for index in range(10000):
            pos, neg = loader.get_batch()
            pos = torch.stack(pos).to(device)
            neg = torch.stack(neg).to(device)

            pos_features = self.model(pos)
            neg_features = self.model(neg)

            optimizer.zero_grad()
            loss = ccl_loss(pos_features, neg_features)
            
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            if index % 50 == 0 and index != 0:
                print('avgloss {}'.format(avg_loss/50))
                avg_loss = 0
            if index % 2000 == 0 and index != 0:
                path = os.path.join(model_path, "{}_{}".format(index, avg_loss))
                torch.save(self.model.state_dict(), path)

if __name__ == "__main__":
    trainer = CCL_trainer()
    trainer.train()
