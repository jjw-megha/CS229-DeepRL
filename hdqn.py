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
		self.actor = neural_network(self.args.nnet)
		self.actor_optimizer = optimizer_spec.constructor(self.actor.parameters(), **optimizer_spec.kwargs)

    def select_move(self, state, goal, goal_value):
		input_vector = np.concatenate([state, goal], axis=2)
		if random.random() < self.actor_epsilon[goal_value]:
			print "Exploring action"
			return torch.IntTensor([random.randrange(self.args.num_actions)])
			#print "Here ------>", self.actor(Variable(torch.from_numpy(vector).float())).data.numpy()
		action_prob = self.actor(Variable(torch.from_numpy(input_vector).type(torch.FloatTensor), volatile=True)).data
        print(action_prob)
        return np.argmax(action_prob)

    def select_goal(self, state):
        return meta_controller.select_goal(state)

    def criticize(self, goal, state):
        man_mask = object_detection.get_man_mask(state)
        return object_detection.getoverlap(man_mask, goal)

	def store(self, experience):
		self.memory.append(experience)

    def _update(self):
		if len(self.memory) < self.batch_size:
			return
		exps = [random.choice(self.memory) for _ in range(self.n_samples)]
		state_vectors = np.squeeze(np.asarray([np.concatenate([exp.state, exp.goal], axis=1) for exp in exps]))

		state_vectors_var = Variable(torch.from_numpy(state_vectors).type(torch.FloatTensor))

		action_batch = np.array([exp.action for exp in exps])
		action_batch_var = Variable(torch.from_numpy(action_batch).long())

		reward_batch = np.array([exp.reward for exp in exps])
		reward_batch_var = Variable(torch.from_numpy(reward_batch).type(torch.FloatTensor))

		done_batch = np.array([exp.done for exp in exps])
		not_done_batch_mask = Variable(torch.from_numpy(1- done_batch).type(torch.FloatTensor))
		#print "state_vectors", state_vectors
		next_state_vectors = np.squeeze(np.asarray([np.concatenate([exp.next_state, exp.goal], axis=1) for exp in exps]))
		next_state_vectors_var = Variable(torch.from_numpy(next_state_vectors).type(torch.FloatTensor))

		try:
			reward_vectors = self.actor(state_vectors_var).gather(1, action_batch_var.unsqueeze(1))
		except Exception as e:
			state_vectors = np.expand_dims(state_vectors, axis=0)
			reward_vectors = self.actor(state_vectors_var).gather(1, action_batch_var.unsqueeze(1))

		try:
			next_state_max_reward = self.target_actor(next_state_vectors_var).detach().max(1)[0]
		except Exception as e:
			next_state_vectors = np.expand_dims(next_state_vectors, axis=0)
			next_state_max_reward = self.target_actor(next_state_vectors_var).detach().max(1)[0]

		next_state_reward_vectors = not_done_batch_mask * next_state_max_reward
		target_Q_values = reward_batch_var + ( self.gamma * next_state_reward_vectors)
		loss = F.smooth_l1_loss(reward_vectors, target_Q_values)
		self.target_actor.load_state_dict(self.actor.state_dict())
		self.actor_optimizer.zero_grad()
		loss.backward()
		for param in self.actor.parameters():
			param.grad.data.clamp_(-1, 1)
		self.actor_optimizer.step()
