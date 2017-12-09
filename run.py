import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import gym
from agent.Hdqn import Hdqn
#from utils.plotting import plot_episode_stats, plot_visited_states
#from utils import plotting
from meta_controller import meta_controller
from object_detection import object_detection
import cv2
import copy 
plt.style.use('ggplot')

class Coach:
    def __init__(self):
        self.env = gym.make('MontezumaRevenge-v4')
        self.env_actions = self.env.unwrapped.get_action_meanings()
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
        self.ale_lives = 6
        self.object_detection = object_detection()

    def learn_subgoal(self):

        goal_mask = self.object_detection.to_grayscale(self.object_detection.downsample(self.goal_mask))
        action = self.agent.select_move(list(self.history)[1:5], goal_mask, self.goal_idx[self.goal])
        print("GOAL", self.goal, str((self.meta.getCurrentState()   , self.env_actions[action])) + "; ")

        next_frame , external_reward, done, info = self.env.step(action)
        # print "Done", done, "Info : ", info['ale.lives']
        if info['ale.lives'] < self.ale_lives:

            self.ale_lives = info['ale.lives']
            print "Agent Died!!!! . Lives left : ", self.ale_lives
            self.meta.update_state('start')
            self.goal = self.meta.getSubgoal()
        cv2.imshow('image', next_frame)
        cv2.waitKey(1)
        # cv2.imshow('image', self.goal_mask)
        # cv2.waitKey(1)
        next_frame_preprocessed = self.object_detection.preprocess(next_frame)
        self.history.append(next_frame_preprocessed)
        if external_reward > 0:
            print "extrinsic_reward for goal", self.goal, " reward:", external_reward

        # print self.goal_mask.shape , next_frame.shape
        intrinsic_reward = self.agent.criticize(self.goal_mask, next_frame)

        goal_reached = (intrinsic_reward > 0)
        if goal_reached:
            print "Goal reached!! ", self.goal
            self.meta.update_state(self.goal)
                
        if len(self.history) == 5:
            exp = self.ActorExperience(copy.deepcopy(list(self.history)[0:4]), goal_mask, action, intrinsic_reward, copy.deepcopy(list(self.history)[1:5]))
            self.agent.store(exp)
        self.agent.update()
        return external_reward, goal_reached, done

    def learn_global(self):
        print "Annealing factor: " + str(self.anneal_factor)
        for num_episode in range(self.num_episodes):
                self.history.clear()
                total_external_reward = 0
                episode_length = 0
                print "\n\n### EPISODE "  + str(num_episode) + "###"
                self.env.reset()
                self.meta = meta_controller()
                done = False
                while not done:
                    frame = self.env.render(mode='rgb_array')
                    frame = self.object_detection.preprocess(frame)
                    self.history.append(frame)
                    self.goal, self.goal_mask = self.meta.getSubgoal()
                    self.stats['goal_selected'][self.goal_idx[self.goal]] += 1
                    print "\nNew Goal: "  + str(self.goal) + "\nState-Actions: "
                    goal_reached = False
                    while not done and not goal_reached:
                        external_reward, goal_reached, done = self.learn_subgoal()
                        
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


def main():
    coach = Coach()
    coach.learn_global()


if __name__ == "__main__":
    main()
