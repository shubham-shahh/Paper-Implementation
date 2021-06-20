import torch
from torch._C import device
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.dropout import Dropout

import torchvision
from torchvision.datasets.mnist import FashionMNIST
import torchvision.transforms as transforms

import torch.nn.functional as F
import torch.optim as optim
from utils import *


VGG_types = {
    'VGG11' : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'VGG13' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

    }

class VGG_net(nn.Module):
    def __init__(self, in_channels=3, num_classes = 1000):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types['VGG16'])

        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)

        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)

        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for index, x in enumerate(architecture):
            if type(x) == int:
                out_channels = x
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)), nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]

        return nn.Sequential(*layers)


def main():
    train_set = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST', train=True, download=True,
            transform= transforms.Compose([transforms.CenterCrop(228, 228),transforms.ToTensor()]))
            #.Normalize(mean, std, inplace=False) is used in VGG paper

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 100)
    network = VGG_net(in_channels=1, num_classes=10)
    network.to(device)
    Train(network, train_loader, device, epochs  = 5, load_model=False)
    #load_checkpoint(network, path = "./checkpoints/my_checkpoint.pth.tar")
    #evaluate(train_loader, network, device)


if __name__ == "__main__":
    main()
































        