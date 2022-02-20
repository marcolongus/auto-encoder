import torch
import torch.nn as nn
import torch.nn.functional as F

# CUDA or CPU implementation
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(f"Neural Network running on {device}")


class Autoencoder(nn.Module):
    ''' Autor encorder  A -> B -> B -> A ''' 
    def __init__(self, n=512):
        super(Autoencoder, self).__init__()
        self.fc1 = nn.Linear(28*28, n)
        self.fc2 = nn.Linear(n, n)
        self.fc3 = nn.Linear(n, 28*28)

    def forward(self, x):
        x = F.relu(F.dropout(self.fc1(x), p=0.1))
        x = F.relu(F.dropout(self.fc2(x), p=0.1))
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
