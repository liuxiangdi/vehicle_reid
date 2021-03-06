import os
import time
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
        self.root_path = "/home/lxd/datasets/VeRi/image_train"
        images = [[int(i[:4]), os.path.join(self.root_path, i)] 
                   for i in os.listdir(self.root_path) if 'jpg' in i]
        images.sort()
        cars_id = []
        ids = []
        
        for index, image in images:
            if index not in ids:
                ids.append(index)
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
    
    def get_contrastive_batch(self, batch_size = 4):
        anchors = []
        simenses = []
        flags = []
        for i in range(batch_size):
            # same id
            flag = 1
            anchor_id, simense_id = None, None
            if random.random() > 0.5:
                anchor_id = random.choice(range(self.id_num))
                simense_id = anchor_id
                flag = 1
            else:
                anchor_id, simense_id = random.sample(range(self.id_num), 2)
                flag = 0
            flags.append(flag)            
            anchor_lis = self.cars_id[anchor_id]
            simense_lis = self.cars_id[simense_id]
            anchor = train_transform(Image.open(random.choice(anchor_lis)))
            simense = train_transform(Image.open(random.choice(simense_lis)))

            anchors.append(anchor)
            simenses.append(simense)

        anchors = torch.stack(anchors)
        simenses = torch.stack(simenses)

        return anchors, simenses, flags


    def get_test_ids(self):
        return range(300)



class VeRI_validation_dataloader():
    def __init__(self):
        super().__init__()
        self.root_path = "/home/lxd/datasets/VeRi/"
        query_path = os.path.join(self.root_path, "image_query")
        self.query_images = [os.path.join(query_path, i) for i in os.listdir(query_path) if ".jpg" in i]

        test_path = os.path.join(self.root_path, "image_test")
        self.test_images = [os.path.join(test_path, i) for i in os.listdir(test_path) if ".jpg" in i]

        self.query_index = 0
        self.test_index = 0

    def get_query_batch(self):
        # 固定batch_size为16
        length = len(self.query_images)
        if self.query_index >= length:
            # end_flag, inputs, image_names
            return True, None, None

        images = []
        car_infos = []
        for image_path in self.query_images[self.query_index:self.query_index+16]:
            images.append(train_transform(Image.open(image_path)))
            car_info = image_path.split('/')[-1][:-4]
            #car_info = _temp.split()
            car_infos.append(car_info)
        images = torch.stack(images)
        
        self.query_index += 16

        return False, images, car_infos
 
    def get_test_batch(self):
        # 固定batch_size为16
        length = len(self.test_images)
        if self.test_index >= length:
            # end_flag, inputs, image_names
            return True, None, None

        images = []
        car_infos = []
        for image_path in self.test_images[self.test_index:self.test_index+16]:
            images.append(train_transform(Image.open(image_path)))
            car_info = image_path.split('/')[-1][:-4]
            #car_info = _temp.split()
            car_infos.append(car_info)
        images = torch.stack(images)
        
        self.test_index += 16
        
        return False, images, car_infos
    

class VeRIdataset_contrastive_train(Dataset):
    def __init__(self):
        self.root_path = "/home/lxd/datasets/VeRi/image_train"
        images = [[int(i[:4]), os.path.join(self.root_path, i)] 
                   for i in os.listdir(self.root_path) if 'jpg' in i]
        images.sort()
        cars_id = []
        ids = []
        
        for index, image in images:
            if index not in ids:
                ids.append(index)
                cars_id.append([])
            cars_id[-1].append(image)
        cars_id = [i for i in cars_id if len(i) >= 4]
        
        self.cars_id = cars_id
        self.id_num = len(self.cars_id)
        print("------------ VeRI Train Dataset : {} class -------------".format(self.id_num))
    
    def __getitem__(self, index):
        flag = 1
        anchor_id, simense_id = None, None
        if random.random() > 0.5:
            anchor_id = random.choice(range(self.id_num))
            simense_id = anchor_id
            flag = 1
        else:
            anchor_id, simense_id = random.sample(range(self.id_num), 2)
            flag = 0
        anchor_lis = self.cars_id[anchor_id]
        simense_lis = self.cars_id[simense_id]
        anchor = train_transform(Image.open(random.choice(anchor_lis)))
        simense = train_transform(Image.open(random.choice(simense_lis)))
        flag = torch.tensor(flag)
        return anchor, simense, flag
    
    def __len__(self):
        return self.id_num


if __name__ == '__main__':
    dataset = VeRIdataset_contrastive_train()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8)
    t = time.time()
    for anchor, simense, flag in dataloader:
        print(flag)
    print(time.time()-t)
