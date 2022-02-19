from modules.train_model import *
from modules.graphics import *

H = 64
EPOCHS = 10
PATH = PATH = f"models/{H}-{EPOCHS}"
model = torch.load(PATH)

test_graph(EPOCHS, H, stop=False, rand_batch=rand_batch, net=model)