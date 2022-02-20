import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torchvision

# CUDA or CPU implementation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Neural Network running on {device}")


class Autoencoder(nn.Module):
    """Autor encorder  A -> B -> B -> A"""

    def __init__(self, n=512):
        super(Autoencoder, self).__init__()
        self.fc1 = nn.Linear(28 * 28, n)
        self.fc2 = nn.Linear(n, n)
        self.fc3 = nn.Linear(n, 28 * 28)


    def forward(self, x):
        x = F.relu(F.dropout(self.fc1(x), p=0.1))
        x = F.relu(F.dropout(self.fc2(x), p=0.1))
        x = self.fc3(x)

        return x



if __name__ == '__main__':

    writer = SummaryWriter("runs/fmnist")
    net = Autoencoder()
    with torch.no_grad():
        x = net(torch.rand(28,28).view(-1, 28*28))
        img_grid = torchvision.utils.make_grid(x.view(28,28)) 
    writer.add_image('fmnist', img_grid)
    writer.add_graph(net, x)
    
    writer.close()
