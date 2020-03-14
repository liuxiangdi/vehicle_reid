import torch
import sys
import time
import os
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from model import Vgg16Net, MobileNet, ResNet50, Vgg11Net, ResNet34
from dataloader import VeRi_dataloader
from losss import BatchHardTripletLoss

################## config ################
model_name = sys.argv[1]
if model_name == "vgg16":
    model = Vgg16Net()
elif model_name == "mobile":
    model = MobileNet()
elif model_name == "mobile":
    model = MobileNet()
elif model_name == "res50":
    model = ResNet50()
elif model_name == "res34":
    model = ResNet34()
elif model_name == "vgg11":
    model = Vgg11Net()
# train/test
mode = sys.argv[2]
dataloader = VeRi_dataloader()
device = torch.device("cuda:0")
date = time.strftime("%m-%d", time.localtime())
model_path = "/home/lxd/checkpoints/" + date
############################################


if not os.path.exists(model_path):
    os.makedirs(model_path)

def trainer_BHTriplet(model, epoch=20000):
    lr = 0.001
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8)
    avg_loss = 0
    criterion = BatchHardTripletLoss()
    for index in range(epoch):
        if index % 10000 == 0:
            lr /= 10
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8)

        batch_inputs = dataloader.get_batch_hard_triplets()
        batch_inputs = batch_inputs.to(device)      

        outputs = model(batch_inputs)
        loss = criterion(outputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
        if index % 2000 == 0 and index != 0:
            path = os.path.join(model_path, "{}_VeRI_{}.pt".format(model_name, index))
            torch.save(self.model.state_dict(), path)
        if index % 100 == 0 and index != 0:
            print('batch {}  avgloss {}'.format(index, avg_loss/100))
            avg_loss = 0

def test():
    pass

def main():
    trainer_BHTriplet()

if __name__ == "__main__":
    main()