import torch
from torch._C import device
import torch.nn as nn

import torchvision
from torchvision.datasets.mnist import FashionMNIST
import torchvision.transforms as transforms

import torch.nn.functional as F
import torch.optim as optim



def get_correct_preds(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def save_checkpoint(state, filename="./checkpoints/my_checkpoint.pth.tar"):
    print("Saving Checkpoint")
    torch.save(state,filename)


def load_checkpoint(network, path):
    print("Loading Checkpoint")
    optimizer = optim.Adam(network.parameters(), lr = 0.01)
    
    checkpoint = torch.load(path)
    network.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    
    #optimizer.load_state_dict(checkpoint['optimizer'])
    #for state in optimizer.state.values():
        #for k, v in state.items():
            #if isinstance(v, torch.Tensor):
                #state[k] = v.cuda()

def Train(network, train_loader, device, epochs=20, load_model = False):

    load_model = load_model


    optimizer = optim.Adam(network.parameters(), lr = 0.01)
    if load_model:
        load_checkpoint(network, path = "./checkpoints/my_checkpoint.pth.tar")

    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
            }
        
        if epoch % 10 == 0:
            save_checkpoint(checkpoint) 

        for batch in train_loader:
            images,labels = batch
            images, labels = images.to(device), labels.to(device)

            preds = network(images)
            loss = F.cross_entropy(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += get_correct_preds(preds, labels)

        print("epoch: ",epoch, "total_correct: ", total_correct, "loss: ", total_loss)




def evaluate(loader, network,device):
    num_correct = 0
    num_samples = 0
    network.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = network(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            accuracy = (num_correct/num_samples)*100
        print(f'accuracy {accuracy}')
