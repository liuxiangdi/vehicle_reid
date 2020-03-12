import torch
import torch.nn as nn
from torchvision import models, transforms

# embedding net for similarity
class MobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        mobileNet = models.mobilenet_v2(pretrained=True)
        feature_map = mobileNet.features
        embedding = mobileNet.classifier
        embedding[1] = nn.Linear(in_features=1280, out_features=128, bias=True)
        self.feature_map = feature_map
        self.embedding = embedding
    
    def forward(self, x):
        x = self.feature_map(x)
        x = x.mean([2, 3])
        x = self.embedding(x)
        x = torch.div(x, torch.norm(x, 2))
        return x
 

class Vgg16Net(nn.Module):
    def __init__(self):
        super(Vgg16Net, self).__init__()
        self.CNN = models.vgg16(pretrained=True).features
        self.FC1 = nn.Linear(7*7*512, 2048)
        self.FC2 = nn.Linear(2048, 128)
    
    def forward(self, x):
        output = self.CNN(x)
        output = output.view(output.size()[0], -1)
        output = self.FC1(output)
        output = nn.functional.relu(output)
        output = self.FC2(output)

        output = torch.div(output, torch.norm(output, 2))
        return output

#class Vgg16_norm(nn.Module):


if __name__ == '__main__':
    #m = CCLNet()
    #m = SiameseNet()
    #m = HardTripletNet()
    m = Vgg16Net()
    print(m)
