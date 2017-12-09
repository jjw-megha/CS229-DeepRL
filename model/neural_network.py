import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os

class neural_network(nn.Module):
	def __init__(self, actions):
		super(neural_network, self).__init__()
		self.conv1 = nn.Conv2d(5, 32, 8, stride=4)
		self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
		self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

		self.bn1 = nn.BatchNorm2d(32)
		self.bn2 = nn.BatchNorm2d(64)
		self.bn3 = nn.BatchNorm2d(64)

		self.fc1 = nn.Linear(2688, 512)
		nn.init.xavier_normal(self.fc1.state_dict()['weight'])
		self.fc2 = nn.Linear(512, actions)
		nn.init.xavier_normal(self.fc2.state_dict()['weight'])


	def forward(self, x):
		# print "in forward"
		# print x.size()
		x = F.relu(self.bn1(self.conv1(x)))
		# print x.size()
		x = F.relu(self.bn2(self.conv2(x)))
		# print x.size()
		x = F.relu(self.bn3(self.conv3(x)))
		# print x.size()
		x = x.view(-1, self.num_flat_features(x))
		# print x.size()
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

	def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
		filepath = os.path.join(folder, filename)
		if not os.path.exists(folder):
			print("Checkpoint Directory does not exist! Making directory {}".format(folder))
			os.mkdir(folder)
		else:
			print("Checkpoint Directory exists! ")
		torch.save({
			'state_dict' : self.state_dict(),
		}, filepath)

	def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
		filepath = os.path.join(folder, filename)
		if not os.path.exists(filepath):
			raise("No model in path {}".format(checkpoint))
		checkpoint = torch.load(filepath)
		self.load_state_dict(checkpoint['state_dict'])

	def mse_loss(self,input, target):
		return torch.sum((input - target) ** 2) / input.data.nelement()
