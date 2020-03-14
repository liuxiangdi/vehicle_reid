import numpy as np


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix

def validation(features, labels):
    """
    features [num x 128, num x 128..]
    """
    features = np.asarray([i.cpu().numpy for i in features])
    dis = pdist(features)

    cmc = []
    for i in range(len(dis)):
        anchor_label = labels[i]
        for rank in range(10):
            _index = np.argmin(dis[i])
            _label = labels(_index)
            if anchor_label == _label:
                cmc.append(rank)
                break
            dis[i][_index] = float("inf")
    rank10 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in cmc:
        rank10[i] += 1/len(features)
    print("CMC {}".format(rank10))

a = np.asarray([1,2,3,4])
b = [a,a,a,a]
c = np.concatenate(a)
print(c)