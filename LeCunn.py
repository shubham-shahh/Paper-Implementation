import torch
from torch._C import device
import torch.nn as nn

import torchvision
from torchvision.datasets.mnist import FashionMNIST
import torchvision.transforms as transforms

import torch.nn.functional as F
import torch.optim as optim
from utils import * 

#LeNet Paper
# http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

#ETL
# 1) Extract - Get the image data from the source
# 2) Transform - Put our data into Tensor form
# 3) Load - Put our data into an object to make it easily accessible

#Pytorch has 2 classes Dataset and DataLoader to process data
#(not necessary for MNIST as it comes with torchvision and does it behind the scenes)




class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = nn.ReLU()

        self.pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2,2))

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, 
        kernel_size=(5,5), stride=(1,1), padding=(2,2))

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, 
        kernel_size=(5,5), stride=(1,1), padding=(0,0))

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, 
        kernel_size=(5,5), stride=(1,1), padding=(0,0))

        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x) # num_examples x 120 x 1 x 1 --> num_examples x 120
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x
        

#Training
#1) Get the batch from the training set
#2) Pass the batch to the Network
#3) calculate the loss (difference between pridicted and the actual value)
#4) calculate the gradient of the loss function w.r.t to the network's weights
#5) Update the weights using gradients to reduce the loss
#6) Repeat setps 1-5 until one Epoch is completed (All batches of the training set)
#7) repeat steps 1-6 for required number of epochs

#Main


def main():
    train_set = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST', train=True, download=True,
            transform= transforms.Compose([transforms.ToTensor()]))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 100)
    network = LeNet()
    network.to(device)
    Train(network, train_loader, device, epochs  = 5, load_model=False)
    #load_checkpoint(network, path = "./checkpoints/my_checkpoint.pth.tar")
    #evaluate(train_loader, network, device)


if __name__ == "__main__":
    main()
























#Test
#x = torch.randn(64,1,32,32)
#model = LeNet()
#print(model(x).shape)
#acc = total_correct/len(train_set)







        