import torch
import torch.nn as nn
import torchvision
import pdb

class Backbone(nn.Module):
    def __init__(self, backbone='resnet50'):
        super(Backbone, self).__init__()

        if backbone == 'resnet18':
            resnet = torchvision.models.resnet.resnet18(pretrained=True)
        elif backbone == 'resnet50':
            resnet = torchvision.models.resnet.resnet50(pretrained=True)
        elif backbone == 'resnet101':
            resnet = torchvision.models.resnet.resnet101(pretrained=True)

        self.block0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        )
        self.block1 = resnet.layer1
        self.block2 = resnet.layer2
        self.block3 = resnet.layer3
        self.block4 = resnet.layer4

    def forward(self, x, returned=[4]):
        blocks = [self.block0(x)]

        blocks.append(self.block1(blocks[-1]))
        blocks.append(self.block2(blocks[-1]))
        blocks.append(self.block3(blocks[-1]))
        blocks.append(self.block4(blocks[-1]))

        out = [blocks[i] for i in returned]

        return out

class new_model(nn.Module):
    def __init__(self,output_layer = None):
        super().__init__()
        self.pretrained = torchvision.models.resnet18(pretrained=True)
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        # pdb.set_trace()
        for i in range(1,len(self.layers)-self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])
        
        self.net = nn.Sequential(self.pretrained._modules)
        self.pretrained = None

    def forward(self,x):
        x = self.net(x)
        return x

class comb_resnet(nn.Module):
    def __init__(self):
        super(comb_resnet,self).__init__()
        self.l1 = new_model(output_layer = 'layer1').eval().cuda()
        self.l2 = new_model(output_layer = 'layer2').eval().cuda()
        self.l3 = new_model(output_layer = 'layer3').eval().cuda()
        self.l4 = new_model(output_layer = 'layer4').eval().cuda()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
    
    def forward(self,img1):
        f1 = self.pool(self.l1(img1)) #.squeeze()
        f2 = self.pool(self.l2(img1)) #.squeeze()
        f3 = self.pool(self.l3(img1)) #.squeeze()
        f4 = self.pool(self.l4(img1)) #.squeeze()
        # pdb.set_trace()
        con = torch.cat((f1,f2,f3,f4),1)
        return con