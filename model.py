import torch
import torch.nn as nn
from torchvision import models, transforms

class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
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
        self.fc1 = nn.Linear(in_features = 2048, out_features = 32, bias = True)
        #self.fc2 = nn.Linear(in_features = 512, out_features = 64, bias = True)
    
    def forward(self, x):
        x = self.feature_map(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.div(x, x.pow(2).sum(1, keepdim=True).sqrt())
        
        return x


class ResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet34(pretrained=True)
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
        self.fc1 = nn.Linear(in_features = 512, out_features = 128, bias = True)
        self.fc2 = nn.Linear(in_features = 128, out_features = 32, bias = True)
    
    def forward(self, x):
        x = self.feature_map(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.div(x, x.pow(2).sum(1, keepdim=True).sqrt())
        return x


class MobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        mobileNet = models.mobilenet_v2(pretrained=True)
        feature_map = mobileNet.features
        embedding = mobileNet.classifier
        embedding[1] = nn.Linear(in_features=1280, out_features=32, bias=True)
        self.feature_map = feature_map
        self.embedding = embedding
        #self.fc = nn.Linear(in_features=512, out_features=128, bias=True)
    
    def forward(self, x):
        x = self.feature_map(x)
        x = x.mean([2, 3])
        x = self.embedding(x)
        #x = nn.functional.relu(x)
        #x = self.fc(x)
        x = torch.div(x, x.pow(2).sum(1, keepdim=True).sqrt())
        
        return x
 

class Vgg16Net(nn.Module):
    def __init__(self):
        super(Vgg16Net, self).__init__()
        self.CNN = models.vgg16(pretrained=True).features
        self.FC1 = nn.Linear(7*7*512, 2048)
        self.FC2 = nn.Linear(2048, 32)
    
    def forward(self, x):
        output = self.CNN(x)
        output = output.view(output.size()[0], -1)
        output = self.FC1(output)
        output = nn.functional.relu(output)
        output = self.FC2(output)
        output = torch.div(output, output.pow(2).sum(1, keepdim=True).sqrt())

        return output


class Vgg11Net(nn.Module):
    def __init__(self):
        super(Vgg11Net, self).__init__()
        self.CNN = models.vgg11(pretrained=True).features
        self.FC1 = nn.Linear(7*7*512, 2048)
        self.FC2 = nn.Linear(2048, 32)
    
    def forward(self, x):
        output = self.CNN(x)
        output = output.view(output.size()[0], -1)
        output = self.FC1(output)
        output = nn.functional.relu(output)
        output = self.FC2(output)

        return output



#class Vgg16_norm(nn.Module):

if __name__ == '__main__':
    #m = CCLNet()
    #m = SiameseNet()
    #m = HardTripletNet()
    m = ResNet()
    print(m)
