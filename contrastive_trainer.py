import torch
import sys
import time
import os
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from model import Vgg16Net, MobileNet, ResNet50, Vgg11Net, ResNet34, AlexNet
from utils import validation
from dataloader import VeRi_dataloader
from losss import BatchHardTripletLoss, contrastive_loss

################## config ################
date = time.strftime("%m-%d", time.localtime())
model_path = "/home/lxd/checkpoints/" + date

model_name = sys.argv[1]
if model_name == "vgg16":
    model = Vgg16Net()
elif model_name == "alexnet":
    model = AlexNet()
elif model_name == "mobile":
    model = MobileNet()
elif model_name == "res50":
    model = ResNet50()
elif model_name == "res34":
    model = ResNet34()
elif model_name == "vgg11":
    model = Vgg11Net()
else:
    print("Moddel Wrong")


gpu = sys.argv[2]
device = torch.device("cuda:{}".format(gpu))
model.to(device)

dataloader = VeRi_dataloader()


############################################


if not os.path.exists(model_path):
    os.makedirs(model_path)

def trainer_Contrastive(model, epoch=50000):
    lr = 0.0002
    optimizer = optim.Adam(model.parameters(), lr=lr)
    avg_loss = 0
    criterion = BatchHardTripletLoss()

    t1, t2, t3 = 0,0,0
    for index in range(epoch):
        if index % 20000 == 0:
            lr /= 10
            optimizer = optim.Adam(model.parameters(), lr=lr)
        _t = time.time()

        anchor, simense, flags = dataloader.get_contrastive_batch()
        t1 += time.time() - _t

        _t = time.time()
        anchor = anchor.to(device)      
        simense = simense.to(device)

        anchor_features = model(anchor)
        simense_features = model(simense)
        
        t2 += time.time() - _t
        _t = time.time()
        loss = contrastive_loss(anchor_features, simense_features, flags)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t3 += time.time() - _t
        avg_loss += loss.item()
        if index % 2000 == 0 and index != 0:
            path = os.path.join(model_path, "{}_Contrastive_VeRI_{}.pt".format(model_name, index))
            torch.save(model.state_dict(), path)
        if index % 50 == 0 and index != 0:
            print('batch {}  avgloss {}'.format(index, avg_loss/50))
            avg_loss = 0

def main():
    trainer_Contrastive(model)

if __name__ == "__main__":
    main()
