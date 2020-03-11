import torch
import torch.nn as nn
from torchvision import models, transforms


# network for color classfication
# use mobile Net
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.feature_map = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(in_features=2048, out_features=8, bias=True)

    def forward(self, x):
        x = self.feature_map(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        #x = self.sigmoid(x)
        return x

# network for color classfication
# use mobile Net
class MobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        mobileNet = models.mobilenet_v2(pretrained=True)
        feature_map = mobileNet.features
        classifier = mobileNet.classifier
        classifier[1] = nn.Linear(in_features=1280, out_features=8, bias=True)
        self.feature_map = feature_map
        self.classifier = classifier

    def forward(self, x):
        x = self.feature_map(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


# embedding net for similarity
class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        mobileNet = models.mobilenet_v2(pretrained=True)
        feature_map = mobileNet.features
        embedding = mobileNet.classifier
        embedding[1] = nn.Linear(in_features=1280, out_features=512, bias=True)
        self.feature_map = feature_map
        self.embedding = embedding
    
    def forward(self, x):
        x = self.feature_map(x)
        x = x.mean([2, 3])
        x = self.embedding(x)
        x = x/(x.pow(2).sum().pow(0.5))
        return x
 
 
class HardTripletNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_net = EmbeddingNet()
     
    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


class CCLNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_net = EmbeddingNet()
     
    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_net = EmbeddingNet()
     
    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2
    
    def get_embedding(self, x):
        return self.embedding_net(x)


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
        return output

if __name__ == '__main__':
    #m = CCLNet()
    #m = SiameseNet()
    #m = HardTripletNet()
    m = Vgg16Net()
    print(m)