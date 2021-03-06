import torch
import sys
import time
import os
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from model import Vgg16Net, MobileNet, ResNet50, Vgg11Net, ResNet34, AlexNet
from utils import validation, cal_dis
from dataloader import VeRi_dataloader, VeRI_validation_dataloader
from losss import BatchHardTripletLoss

################## config ################
device = torch.device("cuda:7")
date = time.strftime("%m-%d", time.localtime())
#date = "03-14"

model_path = "/home/lxd/checkpoints/" + date

model_name = sys.argv[1]
if model_name == "vgg16":
    model = Vgg16Net()
elif model_name == "mobile":
    model = MobileNet()
elif model_name == "alexnet":
    model = AlexNet()
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
loss_name = sys.argv[2]
batch = sys.argv[3]

model.eval()
model.load_state_dict(torch.load("/home/lxd/checkpoints/{}/{}_{}_VeRI_{}.pt".format(date, model_name, loss_name, batch)))
print("model load {}".format(model_name))

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
    mAP = 0
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
            label = test_labels[index]
            cam_id = label[:9]

            #dis_trak_index = get_index(dis_trak, cam_id)
            dis_trak_index = trac_index_map[cam_id]
            if dis_trak[dis_trak_index][0] == name[:9]:
                continue
            
            dis_trak[dis_trak_index][1] = min(dis[index], dis_trak[dis_trak_index][1])
        
        dis_trak.sort(key= lambda x:x[1])
        
        # cal CMC
        rank = 0
        for i in range(10):
            if dis_trak[i][0][:4] == name[:4]:
                ranks[i] += 1
                break
        # cal AP
        ap = 0

        sameid_num = 0
        sameid_num_total = 0
        for i in range(len(dis_trak)):
            if dis_trak[i][0][:4] == name[:4] and dis_trak[i][0] != name[:9]:
                sameid_num_total += 1

        pr = []
        _recall = 0
        for i in range(len(dis_trak)):
            if dis_trak[i][0][:4] == name[:4] and dis_trak[i][0] != name[:9]:
                sameid_num += 1

            precision = sameid_num/(i+1)
            recall = sameid_num/sameid_num_total
            #pr.append([recall, precision])
            if recall != _recall:
                pr.append([recall, precision])
                _recall = recall

        pr.sort()
        for i in range(len(pr)-1):
            ap += (pr[i+1][0] - pr[i][0]) * (pr[i+1][1] + pr[i][1]) / 2
        mAP += ap

        num_control += 1
        if num_control % 100 == 0:
            print("cal rank {}/{}".format(num_control, len(query_features)))
    
    print("mAP {}".format(mAP/len(query_features)))
    for i in range(9):
        ranks[i+1] += ranks[i]
        print("rank{} {}".format(i+1, ranks[i]/len(query_features)))


if __name__ == "__main__":
    test(model)
    #m = get_trajectory()
    #print(m["0002_c002"])
    #print(m["0776_c009"]) 
