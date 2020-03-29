import numpy as np
import time
import torch

def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


def validation(features, labels):
    """
    features [num x 128]
    labels [num x label(int)]
    """
    diss = np.full([len(features), len(features)], float("Inf"), dtype=np.float)
    #diss = np.full([len(features), len(features)], float(0), dtype=np.float)
    for i in range(len(features)):
        if i % 50 == 0:
            print("cal_dis {} / {}".format(i, len(features)))
        for j in range(i+1, len(features)):
            dis = float(np.linalg.norm(features[i] - features[j], 2))
            diss[i][j] = dis
            diss[j][i] = dis
    
    print(diss[0])
    cmc = []
    for i in range(len(diss)):
        anchor_label = labels[i]
        diss[i][i] = float("inf")
        #diss[i][i] = float(0)
        for rank in range(10):
            _index = np.argmin(diss[i])
            #_index = np.argmax(diss[i])
            _label = labels[_index]
            if anchor_label == _label:
                cmc.append(rank)
                break
            diss[i][_index] = float("inf")
            #diss[i][_index] = float(0)
    rank10 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in cmc:
        rank10[i] += 1/len(features)

    for i in range(9):
        rank10[i+1] += rank10[i]
    print("CMC {}".format(rank10))

def cal_dis(anchor, features):
    length = len(features)
    _temp = [anchor for i in range(length)]
    anchor = np.asarray(_temp)

    dis = np.linalg.norm(anchor - features, 2, axis=1)
    return dis

def car_xml():
    train_path = "/home/lxd/datasets/VeRi/train_label.xml"
    train_infos = []
    with open(train_path, 'r') as f:
        for line in f.readlines():
            if "Item " in line:
                _info = line.split("\"")
                image_name = _info[1]
                vehicleID = _info[3]
                cameraID = _info[5]
                colorID = _info[7]
                typeID = _info[9]
                train_infos.append([image_name, vehicleID, cameraID, colorID, typeID])
    print(train_infos[0])
    print(train_infos[-1])
    
    test_path = "/home/lxd/datasets/VeRi/test_label.xml"
    test_infos = []
    with open(test_path, 'r') as f:
        for line in f.readlines():
            if "Item " in line:
                _info = line.split("\"")
                image_name = _info[1]
                vehicleID = _info[3]
                cameraID = _info[5]
                colorID = _info[7]
                typeID = _info[9]
                test_infos.append([image_name, vehicleID, cameraID, colorID, typeID])
    print(test_infos[0])
    print(test_infos[-1])




if __name__ == "__main__":
    """
    test = torch.tensor(np.random.rand(10, 128))
    labels = [1,1,2,2,3,3,4,5,4,5]
    validation(test, labels)
    """

    """
    t = time.time()
    for i in range(10):
        a = np.random.rand(10000, 32)
        b = np.random.rand(32)
        ans = cal_dis(a, b)
    print((time.time() - t)/10)
    """
    car_xml()
