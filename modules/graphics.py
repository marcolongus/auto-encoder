import matplotlib.pyplot as plt
import numpy as np

model_name = 'model-1645055008'

def create_acc_loss_graph(model_name, EPOCHS, H):
	contents = open(f"model-{H}-{EPOCHS}.log","r").read().split('\n')

	times = []
	accuracies = []
	losses = []
	
	val_accs = []
	val_losses = []

	epoch_ticks = []
	
	end_of_epoch = 0
	for c in contents:
		if model_name in c:
			name, epoch, timestamp, loss, val_loss = c.split(",")

			if end_of_epoch != int(epoch):
				end_of_epoch += 1
				epoch_ticks.append(float(timestamp))
					
			times.append(float(timestamp))
			losses.append(float(loss))
			val_losses.append(float(val_loss))

	fig = plt.figure()
	
	ax1 = plt.subplot()
	ax1.set_xlabel('EPOCHS')
	ax1.set_ylim(0, 0.25)
	
	limit = 100
	ax1.set_xticks(epoch_ticks[::limit]) 
	ax1.set_xticklabels([limit*i for i in range(EPOCHS//limit)])
	
	ax1.plot(times, losses, label='Train loss')
	ax1.plot(times, val_losses, label='Test loss')
	ax1.legend()
	
	plt.savefig(f"images/loss-{H}-{EPOCHS}.png")
	plt.close('all')
	del fig
	print('loss graph check')
	#plt.show()

def test_graph(EPOCHS, H, stop, rand_batch, net):

	for i, element in enumerate(rand_batch):
		if i%10==0:
			x = element[0][0].view(28, 28).to('cpu')
			output = net(element[0][0]).view(28, 28).to('cpu')			
			
			plt.subplot(4, 2, 1)
			plt.xticks([]),plt.yticks([])
			plt.imshow(x)
			plt.subplot(4, 2, 2)
			plt.xticks([]),plt.yticks([])
			plt.imshow(output.detach().numpy())

			x = element[0][1].view(28, 28).to('cpu')
			output = net(element[0][1]).view(28, 28).to('cpu')

			plt.subplot(4, 2, 3)
			plt.xticks([]),plt.yticks([])
			plt.imshow(x)
			plt.subplot(4, 2, 4)
			plt.xticks([]),plt.yticks([])
			plt.imshow(output.detach().numpy())		
			

			x = element[0][2].view(28, 28).to('cpu')
			output = net(element[0][2]).view(28, 28).to('cpu')

			plt.subplot(4, 2, 5)
			plt.xticks([]),plt.yticks([])
			plt.imshow(x)
			plt.subplot(4, 2, 6)
			plt.xticks([]),plt.yticks([])
			plt.imshow(output.detach().numpy())		

			plt.tight_layout(pad=0.01, w_pad=0.01, h_pad=0.02)
			
			plt.savefig(f"images/image_{H}-{EPOCHS}-{i}.png")
			plt.close('all')
			#plt.show()
			if stop: break
	
	print('test graph check')


if __name__ == '__main__':
	create_acc_loss_graph(model_name, 20)