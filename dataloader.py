import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

train_transform = transforms.Compose([transforms.Resize((224, 224)), 
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class VeRi_dataloader_triplet():
    # train set 769 cars
    def __init__(self):
        super().__init__()
        self.root_path = "D:\\BaiduNetdiskDownload\\VeRi\\image_train"
        images = [[int(i[:4]), os.path.join(self.root_path, i)] 
                   for i in os.listdir(self.root_path) if 'jpg' in i]
        images.sort()
        cars_id = []
        for index, image in images:
            if index > len(cars_id):
                cars_id.append([])
            cars_id[-1].append(image)
        cars_id = [i for i in cars_id if len(cars_id) >= 4]
        self.cars_id = cars_id
        self.id_num = len(self.cars_id)

    def get_triplet_batch(self, batch_size=4):
        ids = random.sample(range(self.id_num), batch_size+1)
        pos_id, neg_ids = ids[0], ids[1:]
        anchor_images, pos_images, neg_images = [], [], []
        for i in range(batch_size):
            anchor_path, pos_path = random.sample(self.cars_id[pos_id], 2)
            neg_path = random.choice(self.cars_id[neg_ids[i]])
            anchor_images.append(train_transform(Image.open(anchor_path)))
            pos_images.append(train_transform(Image.open(pos_path)))
            neg_images.append(train_transform(Image.open(neg_path)))
        
        anchor_inputs = torch.stack(anchor_images)
        pos_inputs = torch.stack(pos_images)
        neg_inputs = torch.stack(neg_images)
        return anchor_inputs, pos_inputs, neg_inputs
        

if __name__ == '__main__':
    dataloader = VeRi_dataloader_triplet()
    anchor, pos, neg = dataloader.get_triplet_batch()
    
