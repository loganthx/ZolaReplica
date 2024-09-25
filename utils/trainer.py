import torch, torch.nn as nn
import numpy as np, os, matplotlib.pyplot as plt

from tqdm import tqdm

class Trainer:
	def __init__(self, net, config):
		self.config = config
		self.device = config['device']
		self.net = net 

	def train(self, train_dataloader, test_dataloader=None, save=True, load=False, schedule=False, save_logs=True):
		if load and 'weights.pt' in os.listdir():
				self.net.load_state_dict(torch.load('weights.pt', weights_only=True))

		self.net.train()
		loss_fn = eval(f"nn.{self.config['loss']}()")
		print('loss:', loss_fn)
		
		opt = torch.optim.Adam(self.net.parameters(), lr=self.config['lr'])
		if schedule: scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=self.config['lr_step_size'], gamma=self.config['lr_gamma'])

		for epoch in range(1, self.config['epochs'] + 1):
			### EPOCH START
			epoch_loss=[]
			epoch_val_loss=[]
			epoch_val_accuracy=0.0
			self.net.train()

			for inps, lbls in tqdm(train_dataloader):
				lbls = lbls.unsqueeze(dim=1)
				opt.zero_grad()
				outs = self.net(inps.to(self.device))
				loss = loss_fn(outs, lbls.float().to(self.device))
				loss.backward()
				opt.step()
				epoch_loss.append(loss.item())

			if test_dataloader is not None:
				self.net.eval()
				with torch.inference_mode():
					total, rights = 0, 0
					for inps, lbls in test_dataloader:

						total += lbls.shape[0]
						lbls = lbls.unsqueeze(dim=1)
						outs = self.net(inps.to(self.device))

						for i in range(lbls.shape[0]):
							if outs[i].cpu().round().int() == lbls[i].round().int():
								rights += 1
						loss = loss_fn(outs, lbls.to(self.device))
						epoch_val_loss.append(loss.item())

					epoch_val_accuracy = (rights / total) * 100


			### EPOCH END	
			Logs = f'epoch {epoch} loss {np.mean(epoch_loss)} val_loss: {np.mean(epoch_val_loss)} val_acc: {epoch_val_accuracy}%'
			print(Logs)				
			if schedule: scheduler.step()




		if save_logs:
			with open('logs.txt', 'w') as f:
				f.write(Logs)	

		if save:
			torch.save(self.net.state_dict(), 'weights.pt')
			print(f"model weights.pt saved at {os.getcwd()} folder")








