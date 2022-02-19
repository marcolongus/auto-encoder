import torch
import torch.nn as nn
import torch.nn.functional as F

# CUDA or CPU implementation
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f'Running on GPU: {device}')
    print(f'GPU count: {torch.cuda.device_count()}')
else:
    device = torch.device("cpu")
    print(f'Running on CPU:{device}')


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
