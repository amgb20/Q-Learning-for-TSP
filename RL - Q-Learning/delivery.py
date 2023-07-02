# Base Data Science snippet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm_notebook
from scipy.spatial.distance import cdist
import imageio
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

plt.style.use("seaborn-dark")

import sys
sys.path.append("../")
from q_agent import QAgent

class SamplingPointEnvironment(object):
    def __init__(self,n_stops = None,max_box = None, method = "distance", points = None, **kwargs):

        print(f"Initialized Sampling Point Environment with {n_stops} random stops")

        # Initialization

        # added
        self.points = points
        self.n_stops = n_stops
        self.action_space = self.n_stops # action space is the number of stops
        self.observation_space = self.n_stops # observation space is the number of stops as well
        self.max_box = max_box
        self.stops = [] # list of stops that the agent has visited
        self.method = method

        # Generate stops
        self.Generate_Stops()
        self.Generate_Distance_Matrix()
        self.Render()

        # Initialize first point
        self.Reset()

    # genereate the coordinates of the stops
    def Generate_Stops(self):
        xy = np.array(self.points)
        self.x = xy[:, 0]
        self.y = xy[:, 1]

    # generate the distance matrix between the stops using the euclidean distance
    def Generate_Distance_Matrix(self):
        self.q_stops = np.zeros((self.n_stops, self.n_stops))
        for i in range(self.n_stops):
            for j in range(self.n_stops):
                self.q_stops[i][j] = np.sqrt((self.x[i] - self.x[j])**2 + (self.y[i] - self.y[j])**2)
    
    # Drawing the path of the agent
    def Render(self, return_img=False):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        ax.set_title("Delivery Stops")

        # Swap x and y when plotting and invert y (which is plotted on x-axis)
        ax.scatter(self.y, -self.x, c="red", s=50)

        if len(self.stops) > 0:
            # Swap x and y when getting coordinates and invert y
            xy = [self.Get_XY(initial=True)[1], -self.Get_XY(initial=True)[0]]
            xytext = xy[0] - 0.05, xy[1] - 0.1
            ax.annotate("START", xy=xy, xytext=xytext, weight="bold")

        if len(self.stops) > 1:
            # Swap x and y when plotting and invert y
            ax.plot(self.y[self.stops], -self.x[self.stops], c="blue", linewidth=1, linestyle="--")
            xy = [self.Get_XY(initial=False)[1], -self.Get_XY(initial=False)[0]]
            xytext = xy[0] - 0.05, xy[1] - 0.1
            ax.annotate("END", xy=xy, xytext=xytext, weight="bold")

        ax.invert_yaxis()

        plt.xticks([])
        plt.yticks([])

        if return_img:
            fig.canvas.draw_idle()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return image
        else:
            plt.show()
    
    # Reset the environment by clearing all the stops and selecting a random stop as the first stop
    def Reset(self):
        self.stops = []
        first_stop = 0
        self.stops.append(first_stop)
        return first_stop

    # Step function that takes the destination as an input and returns the new state, reward and done
    def Step(self, destination):
        state = self.Get_State()
        new_state = destination
        reward = self.Get_Reward(state, new_state)
        self.stops.append(destination)
        done = len(self.stops) == self.n_stops

        return new_state, reward, done
    
    # returns the most recently bisited stop's index which is the current state of the environment
    def Get_State(self):
        return self.stops[-1]


    def Get_XY(self,initial = False):
        state = self.stops[0] if initial else self.Get_State()
        x = self.x[state]
        y = self.y[state]
        return x,y

    # calculate the reward from moving from one stop to another
    def Get_Reward(self, state, new_state):
        base_reward = self.q_stops[state, new_state]
        return base_reward

def run_episode(env,agent,verbose = 1):

    # Reset the environment
    s = env.Reset()

    # Reset the agent memory
    agent.Reset_Memory()

    # Get the maximum number of stops
    max_Step = env.n_stops
    
    # Initialize the episode reward
    episode_reward = 0

    # added
    total_distance = 0
    
    i = 0
    while i < max_Step:

        # Remember the states
        agent.Remember_State(s)

        # Choose an action
        a = agent.act(s)
        
        # Take the action, and get the reward from environment
        s_next,r,done = env.Step(a)

        # Tweak the reward
        r = -1 * r
        
        if verbose: print(s_next,r,done)
        
        # Update our knowledge in the Q-table
        agent.Train(s,a,r,s_next)
        
        # Update the caches
        episode_reward += r
        s = s_next

        # Add the distance of the taken action
        total_distance -=r
        
        # If the episode is terminated
        i += 1
        if done:
            break
            
    return env,agent,episode_reward, total_distance

class SamplingPointQAgent(QAgent):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.Reset_Memory()

    '''
    Does take into account the current state but also considers the states already visited by the agent.
    The agent avoids revisiting the states that it has already visited by setting their Q-value to -inf.
    '''
    def act(self,s):

        # Get Q Vector
        q = np.copy(self.Q[s,:])

        # Avoid already visited states
        q[self.states_memory] = -np.inf

        if np.random.rand() > self.epsilon:
            a = np.argmax(q)
        else:
            a = np.random.choice([x for x in range(self.actions_size) if x not in self.states_memory])

        return a

    # it allows the agent to remember the states that it has visited and therefore avoid visiting them again
    def Remember_State(self,s):
        self.states_memory.append(s)

    # it allows the agent to forget the states that it has visited and therefore refresh its memory
    def Reset_Memory(self):
        self.states_memory = []

def run_n_episodes(env,agent,name="training.gif",n_episodes=2000,Render_each=100,fps=10):

    # Store the rewards
    rewards = []
    imgs = []
    total_distances = []
    epsilon_values = []

    start_time = time.time()

    # Experience replay
    for i in tqdm_notebook(range(n_episodes)):

        # Run the episode
        env,agent,episode_reward, total_distance = run_episode(env,agent,verbose = 0)
        rewards.append(episode_reward)
        total_distances.append(total_distance)
        epsilon_values.append(agent.epsilon)

        # print('total distance: ', total_distances)

        min_distance = min(total_distances)
        print("Minimum total distance over 1000 episodes: ", min_distance*100000)

        
        if i % Render_each == 0:
            print(agent.Q)
            img = env.Render(return_img = True)
            imgs.append(img)

    # # Show rewards
    # plt.figure(figsize = (15,3))
    # plt.title("Rewards over training")
    # plt.plot(rewards)
    # plt.show()

    elapsed_time = time.time() - start_time
    print("Elapsed time: ", elapsed_time)

    # Create a figure with two subplots: one for rewards, one for total distances
    fig, axs = plt.subplots(3, figsize=(15, 6))

    # Plot rewards
    axs[0].set_title("Rewards over training")
    axs[0].plot(rewards)

    # Plot total distances
    axs[1].set_title("Total distance over training")
    axs[1].plot(total_distances)

    # Plot epsilon values
    axs[2].set_title("Epsilon values over training")
    axs[2].plot(epsilon_values)

    plt.tight_layout()
    plt.show()

    # Save imgs as gif
    imageio.mimsave(name,imgs,duration=100)

    return env,agent