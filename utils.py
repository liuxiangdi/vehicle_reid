import numpy as np
import torch


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix

def validation(features, labels):
    """
    features [num x 128, num x 128..]
    """
    dis = pdist(features)
    print(dis)

    cmc = []
    for i in range(len(dis)):
        anchor_label = labels[i]
        dis[i][i] = float("inf")
        for rank in range(10):
            _index = np.argmin(dis[i])
            _label = labels[_index]
            if anchor_label == _label:
                cmc.append(rank)
                break
            dis[i][_index] = float("inf")
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
