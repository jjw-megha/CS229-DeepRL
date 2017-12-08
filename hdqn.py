import torch
from torch.autograd import Variable
from model.neural_network import neural_network
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque
from copy import deepcopy
import torch.nn.functional as F
from collections import namedtuple
from meta_controller import meta_controller
from object_detection import object_detection

class Hdqn:
    def __init__(self, actor_epsilon = default_actor_epsilon, batch_size=default_batch_size, gamma=default_gamma, optimizer_spec = default_optimizer_spec):
		self.actor_epsilon = actor_epsilon
		self.gamma = default_gamma
		self.batch_size = default_batch_size
		self.memory = deque([], maxlen=self.args.maxlenOfQueue)
        self.args = args
		self.actor = neural_network(self.args.num_actions)
		self.actor_optimizer = optimizer_spec.constructor(self.actor.parameters(), **optimizer_spec.kwargs)


    def select_move(self, state, goal, goal_value):
		input_vector = np.concatenate([state, goal], axis=2)
		if random.random() < self.actor_epsilon[goal_value]:
			print "Exploring action"
			return random.randrange(0, self.args.num_actions)
			#print "Here ------>", self.actor(Variable(torch.from_numpy(vector).float())).data.numpy()
		action_prob = self.actor(Variable(torch.from_numpy(input_vector).type(torch.FloatTensor), volatile=True)).data
        print(action_prob)
        return np.argmax(action_prob)
    '''
    def select_goal(self, frame):
        return meta_controller.select_goal()
    '''
    
    def criticize(self, goal_mask, frame):
        man_mask = object_detection.blob_detect(frame, 'man')
        return object_detection.getoverlap(man_mask, goal)

	def store(self, experience):
		self.memory.append(experience)

    def _update(self):
		if len(self.memory) < self.batch_size:
			return
