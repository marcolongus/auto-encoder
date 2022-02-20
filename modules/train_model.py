from modules.neuralnetwork import *
from modules.data import *
import torch.optim as optim
import time
import random


###############################
#  INIT DATA NETWORK AND DATA #
###############################

MODEL_NAME = f"model-{int(time.time())}"

# Fetch data to GPU memory

train_set = data_loader(1000)[0] 
test_set  = data_loader(100)[1]

cuda_train_set = [(image.view(-1, 28 * 28).to(device), target.to(device)) for image, target in train_set]
rand_batch = [(image.view(-1, 28 * 28).to(device), target.to(device)) for image, target in test_set]


###########################
#  FUNCTIONS DEFINITIONS  #
###########################

def init_netowrk(H=512, lr=0.1):
    net = Autoencoder(H).to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)
    return net, optimizer

def fwd_pass(image, target, net, optimizer, train=False):
    if train:
        net.zero_grad()

    output = net(image)
    loss = F.mse_loss(output, target)

    if train:
        loss.backward()
        optimizer.step()
    return loss

def test_model(net, optimizer, rand_batch=rand_batch):
    images = random.choice(rand_batch)[0]
    target = images

    with torch.no_grad():
        test_loss = fwd_pass(images, target, net, optimizer)

    return test_loss


def train_model(EPOCHS=1, H=512, lr=5):
    
    net, optimizer = init_netowrk(H, lr)

    with open(f"model_logs/model-{H}-{EPOCHS}.log", "w", encoding="utf-8") as f:
        
        #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10 )
        
        for epoch in range(EPOCHS):
            if epoch%11==0: print(f"EPOCH {epoch}.")

            for batch in cuda_train_set:
                images = batch[0]
                target = batch[0]
                
                loss = fwd_pass(images, target, net, optimizer,train=True)
                test_loss = test_model(net, optimizer)
                to_file = float(loss), float(test_loss)
                f.write(f"{MODEL_NAME}, {epoch}, {round(time.time(),3)}, {round(to_file[0],3)}, {round(to_file[1],3)}\n")
                
                #loss_validation = test_loss
            #scheduler.step(loss_validation)

    return net, optimizer


if __name__ == "__main__":
    train_model()
