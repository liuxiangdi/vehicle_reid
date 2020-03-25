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

class VeRi_dataloader():
    # train set 769 cars
    def __init__(self):
        super().__init__()
        #self.root_path = "D:\\BaiduNetdiskDownload\\VeRi\\image_train"
        #self.root_path = "/home/lxd/datasets/VeRi/image_train"
        self.root_path = "/home/lxd/datasets/gt_bbox"
        images = [[int(i[:4]), os.path.join(self.root_path, i)] 
                   for i in os.listdir(self.root_path) if 'jpg' in i]
        images.sort()
        cars_id = []
        for index, image in images:
            if index > len(cars_id):
                cars_id.append([])
            cars_id[-1].append(image)
        cars_id = [i for i in cars_id if len(i) >= 4]
        self.cars_id = cars_id
        self.id_num = len(self.cars_id)
        print("------------ VeRI Train Dataset : {} class -------------".format(self.id_num))

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

    def get_batch_hard_triplets(self, class_num=4, batch_size = 4):
        ids = random.sample(range(self.id_num), class_num)
        images = []
        targets = []
        for _id in ids:
            image_list = self.cars_id[_id]
            image_list = random.sample(image_list, batch_size)
            for image_path in image_list:
                images.append(train_transform(Image.open(image_path)))
                targets.append(_id)
        images = torch.stack(images)
        targets = torch.tensor(targets, dtype = torch.int64)
        return images, targets

    def get_test_batch(self, car_id):
        image_list = self.cars_id[car_id][:4]
        images = []
        targets = []
        for image_path in image_list:
            images.append(train_transform(Image.open(image_path)))
            targets.append(car_id)
        images = torch.stack(images)
        return images, targets

    def get_test_ids(self):
        return range(300)

        

class PersonDataloader():
    def __init__(self, mode):
        self.mode = mode

        if mode == "train":
            self.transform = transforms.Compose([transforms.Resize((224, 224)), 
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        elif mode == "test":
            self.transform = transforms.Compose([transforms.Resize((224, 224)), 
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        images = []
        root_path = "/home/lxd/datasets/gt_bbox"
        _images = [[int(i[:4]), i] for i in os.listdir(root_path) if "jpg" in i]
        _images.sort()
        for image in _images:
            if int(image[0]) > len(images):
                images.append([])
            images[-1].append(os.path.join(root_path, image[1]))
        images = [i for i in images if len(i) >= 4]

        train_num = 1100
        self.train_datas = images[:train_num]
        self.test_datas = images[:400]

        self.test_index = 0

    def get_batch(self, batch_size = 4):
        if self.mode == "train":
            ids = random.sample(range(1100), 5)
            pos_id = ids[0]
            neg_ids = ids[1:]

            pos_path = random.sample(self.train_datas[pos_id], batch_size)
            pos_image = []
            for path in pos_path:
                image = Image.open(path)
                image = self.transform(image)
                pos_image.append(image)
            pos_image = torch.stack(pos_image)
            
            neg_path = [random.choice(self.train_datas[i]) for i in neg_ids]
            neg_image = []
            for path in neg_path:
                image = Image.open(path)
                image = self.transform(image)
                neg_image.append(image)
            neg_image = torch.stack(neg_image)
            
            return pos_image, neg_image

        elif self.mode == "test":
            images_path = self.test_datas[self.test_index][:2]
            images = []
            
            for path in images_path:
                image = Image.open(path)
                image = self.transform(image)
                images.append(image)
            images = torch.stack(images)
            self.test_index += 1
            return images
    
    def get_triplet_batch(self, batch_size=4):
        if self.mode == "train":
            ids = random.sample(range(1100), 5)
            pos_id = ids[0]
            neg_ids = ids[1:]

            pos_path = random.sample(self.train_datas[pos_id], batch_size)
            pos_image = []
            for path in pos_path:
                image = Image.open(path)
                image = self.transform(image)
                pos_image.append(image)
            pos_image = torch.stack(pos_image)
            
            anchor_path = random.sample(self.train_datas[pos_id], batch_size)
            random.shuffle(anchor_path)
            anchor_image = []
            for path in pos_path:
                image = Image.open(path)
                image = self.transform(image)
                anchor_image.append(image)
            anchor_image = torch.stack(anchor_image)
            
            neg_path = [random.choice(self.train_datas[i]) for i in neg_ids]
            neg_image = []
            for path in neg_path:
                image = Image.open(path)
                image = self.transform(image)
                neg_image.append(image)
            neg_image = torch.stack(neg_image)
            
            return anchor_image, pos_image, neg_image

        elif self.mode == "test":
            images_path = self.test_datas[self.test_index][:2]
            images = []
            
            for path in images_path:
                image = Image.open(path)
                image = self.transform(image)
                images.append(image)
            images = torch.stack(images)
            self.test_index += 1
            return images
    


    def get_num(self):
        return 379

if __name__ == '__main__':
    dataloader = VeRi_dataloader()
    anchor, pos, neg = dataloader.get_triplet_batch()
    images = dataloader.get_batch_hard_triplets()
    
