import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class AIC20_dataloader_CCL():
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        if mode == "train":
            self.transform = transforms.Compose([transforms.RandomResizedCrop((224, 224)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        elif mode == "test":
            self.transform = transforms.Compose([transforms.Resize((224, 224)), 
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        
        self.root_path = "/home/lxd/datasets/AIC20_ReID"
        if mode == "train":
            self.image_path = os.path.join(self.root_path, "image_train") 
        else:
            self.image_path = os.path.join(self.root_path, "image_test") 
        self.images = self.select_images(mode)    
        # 测试时保证数据取出
        self.test_index = 0

    def select_images(self, mode):
        fil = None
        if mode == "train":
            fil = os.path.join(self.root_path, "train_track.txt")
        elif mode == "test":
            fil = os.path.join(self.root_path, "test_track.txt")
        images = []
        with open(fil, 'r') as f:
            for line in f.readlines():
                _temp = line.strip().replace('\n', '').split(' ')
                _temp = [os.path.join(self.image_path, i) for i in _temp]
                if len(_temp) > 5 or mode == "test":
                    images.append(_temp)
        return images

    def get_batch(self, batch_size=4):
        if self.mode == "train":
            indexs_num = len(self.images)
            indexs = random.sample(range(indexs_num), batch_size + 1)
            pos_index = indexs[0]
            pos_path = random.sample(self.images[pos_index], batch_size)
            neg_path = []
            for i in range(1, batch_size+1):
                neg_path.append(random.choice(self.images[indexs[i]]))

            pos_image = []
            for path in pos_path:
                image = Image.open(path)
                image = self.transform(image)
                pos_image.append(image)
            pos_image = torch.stack(pos_image)
            
            neg_image = []
            for path in neg_path:
                image = Image.open(path)
                image = self.transform(image)
                neg_image.append(image)
            neg_image = torch.stack(neg_image)
            return pos_image, neg_image

        
        elif self.mode == "test":
            images_path = self.images[self.test_index][:2]
            images = []
            
            for path in images_path:
                image = Image.open(path)
                image = self.transform(image)
                images.append(image)
            images = torch.stack(images)
            self.test_index += 1
            if self.test_index >= len(self.images):
                print("Images fetch out")
            return images

    def get_num(self):
        return len(self.images)


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
    loader = PersonDataloader("train")
    pos, neg = loader.get_batch()
    for i in range(loader.get_num()):
        images = loader.get_batch()
        print(i)
    print("done")
