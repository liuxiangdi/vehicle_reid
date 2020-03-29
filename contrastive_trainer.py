import torch
import sys
import time
import os
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from model import Vgg16Net, MobileNet, ResNet50, Vgg11Net, ResNet34
from utils import validation
from dataloader import VeRi_dataloader
from losss import BatchHardTripletLoss, contrastive_loss

################## config ################
device = torch.device("cuda:1")
date = time.strftime("%m-%d", time.localtime())
model_path = "/home/lxd/checkpoints/" + date

model_name = sys.argv[1]
if model_name == "vgg16":
    model = Vgg16Net()
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
model.to(device)

# train/test
mode = sys.argv[2]
batch = sys.argv[3]
if mode == "test":
    #model.eval()
    model.load_state_dict(torch.load("/home/lxd/checkpoints/{}/{}_Contrastive_VeRI_{}.pt".format(date, model_name, batch)))
    print("model load {}".format(model_name))

dataloader = VeRi_dataloader()


############################################


if not os.path.exists(model_path):
    os.makedirs(model_path)

def trainer_Contrastive(model, epoch=50000):
    lr = 0.0002
    optimizer = optim.Adam(model.parameters(), lr=lr)
    avg_loss = 0
    criterion = BatchHardTripletLoss()
    for index in range(epoch):
        if index % 20000 == 0:
            lr /= 10
            optimizer = optim.Adam(model.parameters(), lr=lr)

        anchor, simense, flags = dataloader.get_contrastive_batch()
        anchor = anchor.to(device)      
        simense = simense.to(device)

        anchor_features = model(anchor)
        simense_features = model(simense)
        
        #loss = criterion(embedding, targets)
        loss = (anchor_features, simense_features, flags)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
        if index % 2000 == 0 and index != 0:
            path = os.path.join(model_path, "{}_BH_VeRI_{}.pt".format(model_name, index))
            torch.save(model.state_dict(), path)
        if index % 50 == 0 and index != 0:
            print('batch {}  avgloss {}'.format(index, avg_loss/50))
            avg_loss = 0

def test(model):
    inputs = []
    labels = []
    for i in dataloader.get_test_ids():
        _inputs, _labels = dataloader.get_test_batch(i)
        _inputs = _inputs.to(device)
        _features = model(_inputs).cpu().detach().numpy()
        for idx in range(len(_features)):
            inputs.append(_features[idx])
        labels += _labels
    validation(inputs, labels)

def main():
    if mode == "train":
        trainer_Contrastive(model)
    elif mode == "test":
        test(model)

if __name__ == "__main__":
    main()