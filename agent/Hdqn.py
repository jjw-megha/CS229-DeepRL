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
from dotdict import dotdict

default_args = dotdict({
	'actor_epsilon': 0.9,
	'gamma':0.9,
	'batch_size':100,
	'num_actions':18,
	'target_update' : 10000, #Number of iterations for annealing
	'checkpoint' :'checkpoint1',
	'maxlenOfQueue': 50000,
	'num_actions':18
})

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])
optimizer_spec = OptimizerSpec(constructor=optim.RMSprop,
	kwargs=dict(lr=0.00025, alpha=0.95, eps=1e-06),)

class Hdqn:
	def __init__(self, args=default_args):
		self.num_actions = args.num_actions
		self.object_detection = object_detection()
		self.actor_epsilon = {0:0.9,1:0.9,2:0.9,3:0.9,4:0.9}
		self.gamma = args.gamma
		self.batch_size = args.batch_size
		self.memory = deque([], maxlen=args.maxlenOfQueue)
		self.actor = neural_network(args.num_actions)
		self.target_actor = neural_network(args.num_actions)
		self.actor_optimizer = optimizer_spec.constructor(self.actor.parameters(), **optimizer_spec.kwargs)
		self.target_update = args.target_update
		self.steps_since_last_update_target = 0
		self.update_number = 0
		self.checkpoint = default_args.checkpoint

	def select_move(self, state, goal, goal_value):
		
		if random.random() < self.actor_epsilon[goal_value] or len(state) < 4:	
			print "Exploring action"
			return random.randrange(0, self.num_actions)
			#print "Here ------>", self.actor(Variable(torch.from_numpy(vector).float())).data.numpy()

		processed_frames = []
		for frame in state:
			frame = self.object_detection.get_game_region(frame)
			processed_frames.append(self.object_detection.preprocess(frame))
		processed_frames.append(self.object_detection.preprocess(goal))
		input_vector = np.concatenate(processed_frames, axis=2)
		print input_vector.shape
		action_prob = self.actor(Variable(torch.from_numpy(input_vector).type(torch.FloatTensor), volatile=True)).data
		print(action_prob)
		return np.argmax(action_prob)
	'''
	def select_goal(self, frame):
		return meta_controller.select_goal()
	'''

	def criticize(self, goal_mask, frame):
		return self.object_detection.get_overlap(frame, goal_mask)

	def store(self, experience):
		self.memory.append(experience)

	def update(self):

		if len(self.memory) < self.batch_size:
			return

		self.update_number += 1

		exps = [random.choice(list(self.memory)) for _ in range(self.batch_size)]

		state_vectors = np.squeeze(np.asarray([np.concatenate([exp.state, exp.goal], axis=2) for exp in exps]))
		state_vectors_var = Variable(torch.from_numpy(state_vectors).type(torch.FloatTensor))

		action_batch = np.array([exp.action for exp in exps])
		action_batch_var = Variable(torch.from_numpy(action_batch).long())

		reward_batch = np.array([exp.reward for exp in exps])
		reward_batch_var = Variable(torch.from_numpy(reward_batch).type(torch.FloatTensor))
		#print "state_vectors", state_vectors
		next_state_vectors = np.squeeze(np.asarray([np.concatenate([exp.next_state, exp.goal], axis=2) for exp in exps]))
		next_state_vectors_var = Variable(torch.from_numpy(next_state_vectors).type(torch.FloatTensor))

		current_Q_values = self.actor(state_vectors_var)
		next_state_Q_values = self.target_actor(next_state_vectors_var)

		target_Q_values = reward_batch_var + (self.gamma * next_state_Q_values)
		criterion = nn.MSELoss()
		loss = criterion(current_Q_values, target_Q_values)

		self.actor_optimizer.zero_grad()
		loss.backward()
		for param in self.actor.parameters():
			param.grad.data.clamp_(-1, 1)
		self.actor_optimizer.step()

		self.actor.save_checkpoint(self.checkpoint , 'checkpoint_'+self.update_number+'.pth.tar')
		if self.steps_since_last_update_target == self.target_update:
			# Update target
			self.target_actor.load_checkpoint(self.checkpoint , 'checkpoint_'+self.update_number+'.pth.tar')
			self.steps_since_last_update_target = 0
		else:
			self.steps_since_last_update_target += 1
