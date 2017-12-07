import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class neural_network(nn.Module):
	def __init__(self, actions):
		super(neural_network, self).__init__()
		self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
		self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
		self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
		self.fc1 = nn.Linear(64 * 3 * 3, 512)
		self.fc2 = nn.Linear(512, actions)
		

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))
        x = self.fc2(x)
		return x

	def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features