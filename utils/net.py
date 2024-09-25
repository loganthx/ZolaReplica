import torch, torch.nn as nn, torch.nn.functional as F



class Block(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.relu = nn.ReLU()
		self.conv = nn.Conv2d(in_channels, out_channels, 2)
		self.pool = nn.MaxPool2d(2)

	def forward(self, x):
		return self.pool(self.relu(self.conv(x)))



class NET(nn.Module):
	def __init__(self):
		super().__init__()
		self.block1 = Block(3, 5)
		self.block2 = Block(5, 5)

		self.fc1 = nn.Linear(5*24*24, 32)
		self.fc2 = nn.Linear(32, 16)
		self.fc3 = nn.Linear(16, 1)


	def forward(self, x):
		x = self.block1(x)
		x = self.block2(x)
		x = x.view(-1, 5*24*24)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.sigmoid(self.fc3(x))

		return x
