import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import gym
from agent.Hdqn import Hdqn
from utils.plotting import plot_episode_stats, plot_visited_states
from utils import plotting
from meta_controller import meta_controller
plt.style.use('ggplot')

class Coach:
    def __init__(self):
        self.env = gym.make('MontezumaRevenge-v4')
        self.env_actions = env.unwrapped.get_action_meanings()
        self.agent = Hdqn()
        self.goal = ''
        self.goal_mask = []
        self.meta = meta_controller()
        self.history = deque([], maxlen = 5)
        self.num_episodes = 1000
        self.anneal_factor = (1.0-0.1)/self.num_episodes
        self.goal_idx = {'ladder1':0,'ladder2':1,'ladder3':2,'key':3 ,'door2':4}
        self.ActorExperience = namedtuple("ActorExperience", ["state", "goal", "action", "reward", "next_state"])
        self.stats = {'episode_rewards': np.zeros(self.num_episodes) , 'episode_length' : np.zeros(self.num_episodes), 'goal_selected': np.zeros(5), 'goal_success': np.zeros(5)}
        self.anneal_threshold = 0.9

    def learn_subgoal(self):

        action = agent.select_move(self.history, self.goal_mask, self.goal_idx[goal])
        print(str((self.meta.get_state, self.env_actions(action))) + "; ")
        next_frame, external_reward, done, _ = self.env.step(action)
        self.history.append(next_frame)
        if external_reward > 0:
            print "extrinsic_reward for goal", self.goal, " reward:", external_reward
        intrinsic_reward = agent.criticize(self.goal_mask, next_frame)
        goal_reached = (intrinsic_reward > 0)
        if goal_reached:
            agent.goal_success[self.goal] += 1
            print "Goal reached!! "
        exp = self.ActorExperience(self.history[0:4], self.goal_mask, action, intrinsic_reward, self.history[1:5])
        agent.store(exp)
        agent.update()
        return external_reward, goal_reached

    def learn_global(self):
        print "Annealing factor: " + str(anneal_factor)
        for num_episode in range(self.num_episodes):
                self.history.clear()
                total_external_reward = 0
                episode_length = 0
                print "\n\n### EPISODE "  + str(num_episode) + "###"
                env.reset()
                self.meta = meta_controller()
                done = False
                while not done:
                    frame = env.render(mode='rgb_array')
                    self.history.append(frame)
                    self.goal, self.goal_mask = meta.getSubgoal()
                    self.stats['goal_selected'][self.goal_idx[self.goal]] += 1
                    print "\nNew Goal: "  + str(self.goal) + "\nState-Actions: "
                    goal_reached = False
                    while not done and not goal_reached:
                        external_reward, goal_reached = learn_subgoal()
                        self.meta.update_state(self.goal)
                        total_external_reward += external_reward
                        episode_length += 1
                        if goal_reached:
                            self.stats['goal_success'][self.goal_idx[self.goal]] += 1
                #Annealing
                self.stats['episode_rewards'][num_episode] = total_external_reward
                self.stats['episode_length'][num_episode] = episode_length
                for goal in self.goal_idx.keys():
                    avg_success_rate = self.stats['goal_success'][self.goal_idx[goal]] / self.stats['goal_selected'][self.goal_idx[goal]]
                    if avg_success_rate < self.anneal_threshold:
                        self.agent.actor_epsilon[self.goal_idx[goal]] -= self.anneal_factor
                        self.agent.actor_epsilon[self.goal_idx[goal]] = max(0.1, self.agent.actor_epsilon[self.goal_idx[goal]])
                    else:
                        self.agent.actor_epsilon[self.goal_idx[goal]] = 0.1
                    print "actor_epsilon " + str(goal) + ": " + str(self.agent.actor_epsilon[self.goal_idx[goal]])


if __name__ == "__main__":
    main()
