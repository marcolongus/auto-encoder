import matplotlib.pyplot as plt
import numpy as np

H_list = [64, 128, 256, 512]
epoch_list = [10000]
	
def model_plot(H_list=H_list, epoch_list=epoch_list):

	fig = plt.figure()
	H=128
	EPOCHS=10000
	
	for H in H_list:
		PATH = f"model_logs/model-{H}-{EPOCHS}.log"
		contents = open(PATH,"r").read().split('\n')

		times = []
		losses = []	
		val_losses = []

		epoch_ticks = []
		end_of_epoch = 0

		for i, c in enumerate(contents):
			if i%100==0:
				try:
					name, epoch, timestamp, loss, val_loss = c.split(",")
				except:
					pass
				if end_of_epoch != int(epoch):
					end_of_epoch += 1
					epoch_ticks.append(float(timestamp))
							
				times.append(float(timestamp))
				losses.append(float(loss))
				val_losses.append(float(val_loss))

		normalize_time = np.array([i for i in range(len(times))])
		print(len(times))
				
		ax1 = plt.subplot()
		ax1.set_xlabel('EPOCHS')
		#ax1.set_ylim(0, 0.1)
				
		ax1.set_xticks(normalize_time[::600]) 
		ax1.set_xticklabels([i*1000 for i in range(10)])
					
		ax1.semilogx(normalize_time, val_losses, label=f'Test loss {H}')
		ax1.semilogx(normalize_time, losses, color='black')
			
		ax1.legend()

	
	plt.savefig(f"loss-comparison-semilogx.png")
	plt.show()


model_plot()