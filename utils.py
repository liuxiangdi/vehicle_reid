import numpy as np
import torch

def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix

def validation(features, labels):
    """
    features [num x 128]
    """
    diss = np.full([len(features), len(features)], float("Inf"), dtype=np.float)
    for i in range(len(features)):
        if i % 100 == 0:
            print("cal_dis {} / {}".format(i, len(features)))
        for j in range(i+1, len(features)):
            dis = float(np.linalg.norm(features[i][0]-features[j][0], 2))
            diss[i][j] = dis
            diss[j][i] = dis
    
    cmc = []
    for i in range(len(diss)):
        anchor_label = labels[i]
        diss[i][i] = float("inf")
        for rank in range(10):
            _index = np.argmin(dis[i])
            _label = labels[_index]
            if anchor_label == _label:
                cmc.append(rank)
                break
            diss[i][_index] = float("inf")
    rank10 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in cmc:
        rank10[i] += 1/len(features)

    for i in range(9):
        rank10[i+1] += rank10[i]
    print("CMC {}".format(rank10))


if __name__ == "__main__":
    test = torch.tensor(np.random.rand(10, 128))
    labels = [1,1,2,2,3,3,4,5,4,5]
    validation(test, labels)
