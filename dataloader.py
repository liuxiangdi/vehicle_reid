import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


class CClDataLoader():
    def __init__(self, mode):
        super().__init__()

        self.transform = transforms.Compose([transforms.Resize((224, 224)), 
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.root_path = "D:\\Workspace\\assests\\AIC20_ReID"
        if mode == "train":
            self.image_path = os.path.join(self.root_path, "image_train") 
        else:
            self.image_path = os.path.join(self.root_path, "image_test") 
        self.images = self.select_images(mode)    

    def select_images(self, mode):
        fil = None
        if mode == "train":
            fil = os.path.join(self.root_path, "train_track.txt")
        else:
            fil = os.path.join(self.root_path, "test_track.txt")
        images = []
        with open(fil, 'r') as f:
            for line in f.readlines():
                _temp = line.strip().replace('\n', '').split(' ')
                _temp = [os.path.join(self.image_path, i) for i in _temp]
                if len(_temp) > 5:
                    images.append(_temp)
        return images

    def get_batch(self, batch_size=4):
        indexs_num = len(self.images)
        pos_index, neg_idx = random.sample(range(indexs_num), 2)
        pos_path = random.sample(self.images[pos_index], batch_size)
        neg_path = random.sample(self.images[neg_idx], batch_size)

        pos_image = []
        for path in pos_path:
            image = Image.open(path)
            image = self.transform(image)
            pos_image.append(image)
        
        neg_image = []
        for path in pos_path:
            image = Image.open(path)
            image = self.transform(image)
            neg_image.append(image)
        return pos_image, neg_image


if __name__ == '__main__':
    loader = CClDataLoader("train")
    pos, neg = loader.get_batch()
    print(pos)


