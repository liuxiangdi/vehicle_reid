import torch
import sys
import time
import os
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from model import Vgg16Net, MobileNet, ResNet50, Vgg11Net, ResNet34
from utils import validation, cal_dis
from dataloader import VeRi_dataloader, VeRI_validation_dataloader
from losss import BatchHardTripletLoss

################## config ################
device = torch.device("cuda:2")
date = time.strftime("%m-%d", time.localtime())
#date = "03-14"

model_path = "/home/lxd/checkpoints/" + date

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
else:
    print("Moddel Wrong")
model.to(device)

# train/test
mode = sys.argv[2]
batch = sys.argv[3]
if mode == "test":
    model.eval()
    model.load_state_dict(torch.load("/home/lxd/checkpoints/{}/{}_BH_VeRI_{}.pt".format(date, model_name, batch)))
    print("model load {}".format(model_name))

dataloader = VeRi_dataloader()


############################################


if not os.path.exists(model_path):
    os.makedirs(model_path)

def trainer_BHTriplet(model, epoch=50000):
    lr = 0.0002
    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    avg_loss = 0
    criterion = BatchHardTripletLoss()
    for index in range(epoch):
        if index % 20000 == 0:
            lr /= 10
            optimizer = optim.Adam(model.parameters(), lr=lr)

        batch_inputs, targets = dataloader.get_batch_hard_triplets()
        batch_inputs = batch_inputs.to(device)      
        targets = targets.to(device)

        embedding = model(batch_inputs)
        
        loss = criterion(embedding, targets)
        
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


#def trajectory_reader():
#    trajectory_path = "/home/lxd/datasets/VeRi/test_track_VeRi.txt"

def get_trajectorys():
    path = "/home/lxd/datasets/VeRi/image_test"
    images = os.listdir(path)
    images.sort()
    id_cams = []
    trajectorys = []
    for image in images:
        if image[:9] not in id_cams:
            id_cams.append(image[:9])
            trajectorys.append([])
        trajectorys[-1].append(image[:-4])

    trajectory_map = {}
    for id_cam, traj in zip(id_cams, trajectorys):
        trajectory_map[id_cam] = traj
    return trajectory_map


def get_index(dis_trak, cam_id):
    for index, i in enumerate(dis_trak):
        if i[0] == cam_id:
            return index

def test(model):
    inputs = []
    query_features = []
    query_labels = []
    test_features = []
    test_labels = []
    
    trajectorys = get_trajectorys()
    dataloader = VeRI_validation_dataloader()

    # 计算query集中的features
    print("cal qeury features ...")
    t = time.time()
    while True:
        end_flag, _inputs, car_infos = dataloader.get_query_batch()
        if end_flag:
            break
        _inputs = _inputs.to(device)
        _features = model(_inputs).cpu().detach().numpy()
        for i in range(len(car_infos)):
            query_labels.append(car_infos[i])
            query_features.append(_features[i])

    print("cost time: {}".format(time.time() - t))
    #print(query_labels)
    print("cal test features ...")
    
    # 计算test集中的features
    t = time.time()
    _test_num = 0
    while True:
        end_flag, _inputs, car_infos = dataloader.get_test_batch()
        if end_flag:
            break
        _inputs = _inputs.to(device)
        _features = model(_inputs).cpu().detach().numpy()
        for i in range(len(_features)):
            test_labels.append(car_infos[i])
            test_features.append(_features[i])
        _test_num += 1
        #if _test_num == 100:
        #    break
    test_feature = np.asarray(test_features)
    print("cost time: {}".format(time.time() - t))

    
    ranks = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    num_control = 0
    for feature, name in zip(query_features, query_labels):
        dis_trak = []
        trac_index_map = {}
        for _i, key in enumerate(list(trajectorys.keys())):
            # dis [id_cam, 距离]
            dis_trak.append([key, float("inf")])
            trac_index_map[key] = _i
        # 对于每一个query，计算与所有test集距离
        dis = cal_dis(feature, test_features)
        
        for index in range(len(dis)):
            _t = time.time()
            label = test_labels[index]
            cam_id = label[:9]

            #dis_trak_index = get_index(dis_trak, cam_id)
            dis_trak_index = trac_index_map[cam_id]
            if dis_trak[dis_trak_index][0] == name[:9]:
                continue
            
            dis_trak[dis_trak_index][1] = min(dis[index], dis_trak[dis_trak_index][1])
        
        dis_trak.sort(key= lambda x:x[1])
        rank = 0
        for i in range(10):
            if dis_trak[i][0][:4] == name[:4]:
                ranks[i] += 1
                break

        num_control += 1
        if num_control % 100 == 0:
            print("cal rank {}/{}".format(num_control, len(query_features)))
        
    for i in range(9):
        ranks[i+1] += ranks[i]
        print("rank{} {}".format(i+1, ranks[i]/len(query_features)))
   # save_


def main():
    if mode == "train":
        trainer_BHTriplet(model)
    elif mode == "test":
        test(model)

if __name__ == "__main__":
    main()
    #m = get_trajectory()
    #print(m["0002_c002"])
    #print(m["0776_c009"])
