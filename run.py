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
import pickle as pkl
import sys
plt.style.use('ggplot')


     
class Coach:
    def __init__(self):
        self.env = gym.make('MontezumaRevenge-v4')
        self.env_actions = self.env.unwrapped.get_action_meanings()
        self.agent = Hdqn(sys.argv[1])
        self.goal = ''
        self.goal_mask = []
        self.meta = meta_controller()
        self.history = deque([], maxlen = 5)
        self.num_episodes = 5000
        self.anneal_factor = (1.0-0.1)/self.num_episodes
        self.ActorExperience = namedtuple("ActorExperience", ["state", "goal", "action", "reward", "next_state", "done"])
        self.stats =  {'episode_rewards': np.zeros(self.num_episodes) , 'episode_length' : np.zeros(self.num_episodes), 'goal_selected': {}, 'goal_success':{}}  
        self.anneal_threshold = 0.8
        self.ale_lives = 6
        self.object_detection = object_detection()
        self.initial_p = 1
        self.final_p = 0.1
        self.schedule_timesteps = 50000
        self.time_steps = 0

    def learn_subgoal(self):

        goal_mask = self.object_detection.to_grayscale(self.object_detection.downsample(self.goal_mask))
        action = self.agent.select_move(list(self.history)[1:5], goal_mask, self.goal)
        print("GOAL", self.goal, str((self.meta.getCurrentState()   , self.env_actions[action])) + "; ")

        next_frame , external_reward, done, info = self.env.step(action)
        print "Done", done, "Info : ", info['ale.lives'], "ale_lives", self.ale_lives
        if info['ale.lives'] < self.ale_lives:
            self.ale_lives = info['ale.lives']
            print "Agent Died!!!! . Lives left : ", self.ale_lives
            self.meta.update_state('start')           
            self.goal, self.goal_mask = self.meta.getSubgoal() 
            self.stats['goal_selected'][self.goal] += 1

        next_frame_preprocessed = self.object_detection.preprocess(next_frame)
        self.history.append(next_frame_preprocessed)
        if external_reward > 0:
            print "extrinsic_reward for goal", self.goal, " reward:", external_reward
            print "Collected Key!!!"
            self.meta.got_key()

        intrinsic_reward = self.agent.criticize(self.goal_mask, next_frame)
        print("Intrinsic Reward", intrinsic_reward)
        goal_reached = (intrinsic_reward > 0)
        if goal_reached:
            print "Goal reached!! ", self.goal
            self.meta.update_state(self.goal)
                
        if len(self.history) == 5:
            exp = self.ActorExperience(copy.deepcopy(list(self.history)[0:4]), goal_mask, action, intrinsic_reward, copy.deepcopy(list(self.history)[1:5]), goal_reached)
            self.agent.store(exp)
        self.agent.update()
        return external_reward, goal_reached, done

    def learn_global(self):
        for goal in self.agent.actor_epsilon.keys():
            if self.goal not in self.stats['goal_selected']:
                self.stats['goal_selected'][goal] = 0
                self.stats['goal_success'][goal] = 0
        print "Annealing factor: " + str(self.anneal_factor)
        for num_episode in range(self.num_episodes):
                self.history.clear()
                self.ale_lives = 6
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
                    self.stats['goal_selected'][self.goal] += 1
                    print "\nNew Goal: "  + str(self.goal) + "\nState-Actions: "
                    goal_reached = False
                    while not done and not goal_reached:
                        self.time_steps += 1
                        external_reward, goal_reached, done = self.learn_subgoal()
                        
                        total_external_reward += external_reward
                        episode_length += 1
                        if goal_reached:
                            self.stats['goal_success'][self.goal] += 1
                #Annealing
                self.stats['episode_rewards'][num_episode] = total_external_reward
                self.stats['episode_length'][num_episode] = episode_length
                for goal in self.agent.actor_epsilon.keys():
                    if goal not in self.stats['goal_selected']:
                        self.stats['goal_selected'][goal] = 0
                        self.stats['goal_success'][goal] = 0

                    if self.stats['goal_selected'][goal] > 0:
                        print("Success Rate", self.stats['goal_success'][goal], self.stats['goal_selected'][goal], goal)
                        avg_success_rate = self.stats['goal_success'][goal] / self.stats['goal_selected'][goal]
                        if avg_success_rate < self.anneal_threshold or self.stats['goal_selected'][goal] < 100:
                            self.agent.actor_epsilon[goal] -= self.anneal_factor
                        else:
                            self.agent.actor_epsilon[goal] = 0.1

                        self.agent.actor_epsilon[goal] = max(0.1, self.agent.actor_epsilon[goal])
                        print "actor_epsilon " + str(goal) + ": " + str(self.agent.actor_epsilon[goal])
                if num_episode % 2:
                    pkl.dump(self.stats, open(sys.argv[1]+"stats.pkl", 'wb'))
                


def main():

    coach = Coach()
    coach.learn_global()
    self.agent.actor.save_checkpoint(self.checkpoint, 'checkpoint_final.pth.tar')



if __name__ == "__main__":
    main()
