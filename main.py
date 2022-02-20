from modules.train_model import *
from modules.graphics import *

def main(EPOCHS=20, H=512, lr=0.1, rand_batch=rand_batch):
	
	net, optimizer = train_model(EPOCHS, H, lr)
	create_acc_loss_graph(MODEL_NAME, EPOCHS, H)
	test_graph(EPOCHS, H, stop=False, rand_batch=rand_batch, net=net)
	
	print("\n Model's state_dict:")
	for param_tensor in net.state_dict():
		print(param_tensor, "\t", net.state_dict()[param_tensor].size(), "\n")

	PATH = f"models/{H}-{EPOCHS}"
	torch.save(net, PATH)

if __name__ == '__main__':

	H_list = [64, 128, 256, 512]
	epoch_list = [1]
	for layers in H_list:
		for element in epoch_list:
			main(element, layers)
	

		



	
		
		

		


