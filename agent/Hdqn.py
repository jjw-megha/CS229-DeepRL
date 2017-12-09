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
import copy

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
		# print len(state)
		processed_frames = state
		processed_frames.append(goal)
		
		input_vector = np.concatenate(processed_frames, axis=2)
		# print input_vector.shape
		h_, w_, n_ = input_vector.shape
		input_vector = input_vector.reshape((-1, n_, h_, w_))
		# input_vector = np.expand_dims(input_vector,)
		# print input_vector.shape
		action_prob = self.actor(Variable(torch.from_numpy(input_vector).type(torch.FloatTensor), volatile=True)).data

		# print "Action_pro", action_prob
		# raw_input()
		# print(action_prob)
		return np.argmax(action_prob)
	'''
	def select_goal(self, frame):
		return meta_controller.select_goal()
	'''

	def criticize(self, goal_mask, frame):
		frame = self.object_detection.get_game_region(frame)
		return self.object_detection.get_overlap(frame, goal_mask)

	def store(self, experience):
		self.memory.append(experience)

	def update(self):

		if len(self.memory) < self.batch_size:
			return

		self.update_number += 1

		exps = [random.choice(list(self.memory)) for _ in range(self.batch_size)]
		
		# print exps[0].state[0].shape , exps[0].goal.shape
		histories = []
		histories_next_state = []
		for exp in exps:
			state = copy.deepcopy(exp.state)
			state.append(copy.deepcopy(exp.goal))
			state = np.concatenate(state, axis=2)
			
			h_, w_, n_ = state.shape
			state = state.reshape(( -1, n_, h_, w_))
			histories.append(np.array(state))

			next_state = copy.deepcopy(exp.next_state)
			next_state.append(copy.deepcopy(exp.goal))
			next_state = np.concatenate(next_state, axis=2)
			h_, w_, n_ = next_state.shape
			next_state = next_state.reshape((-1, n_ , h_, w_))
			histories_next_state.append(np.array(next_state))

		
		state_vectors = np.squeeze(np.array(histories, dtype = np.uint8))
		# print state_vectors.shape
		state_vectors_var = Variable(torch.from_numpy(state_vectors).type(torch.FloatTensor))

		action_batch = np.array([exp.action for exp in exps])
		action_batch_var = Variable(torch.from_numpy(action_batch).long())

		reward_batch = np.array([exp.reward for exp in exps])
		reward_batch_var = Variable(torch.from_numpy(reward_batch).type(torch.FloatTensor))
		#print "state_vectors", state_vectors
		next_state_vectors = np.squeeze(np.asarray(histories_next_state))
		next_state_vectors_var = Variable(torch.from_numpy(next_state_vectors).type(torch.FloatTensor))


		current_Q_values = torch.max(self.actor(state_vectors_var),dim=1)[0]
		next_state_Q_values = torch.max(self.target_actor(next_state_vectors_var),dim=1)[0]

		# print reward_batch_var.size(), next_state_Q_values.size()
		target_Q_values = reward_batch_var + (self.gamma * next_state_Q_values)
		print target_Q_values.size(), current_Q_values.size()
		criterion = self.actor.mse_loss
		loss = criterion(current_Q_values, target_Q_values)

		self.actor_optimizer.zero_grad()
		loss.backward()
		for param in self.actor.parameters():
			param.grad.data.clamp_(-1, 1)
		self.actor_optimizer.step()

		self.actor.save_checkpoint(self.checkpoint , 'checkpoint_'+str(self.update_number)+'.pth.tar')
		if self.steps_since_last_update_target == self.target_update:
			# Update target
			self.target_actor.load_checkpoint(self.checkpoint , 'checkpoint_'+str(self.update_number)+'.pth.tar')
			self.steps_since_last_update_target = 0
		else:
			self.steps_since_last_update_target += 1
