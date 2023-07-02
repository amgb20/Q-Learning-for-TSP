import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import imageio
import sys

from tqdm import tqdm_notebook
from q_agent import QAgent

plt.style.use("seaborn-dark")
sys.path.append("../")

class SamplingPointEnvironment(object):
    def __init__(self,stop_count = None,max_limit = None, points = None):

        print(f"Initialized Sampling Point Environment with {stop_count} random visited_stops")

        # Initialization

        # added
        self.points = points
        self.stop_count = stop_count
        self.action_space = self.stop_count # action space is the number of visited_stops
        self.observation_space = self.stop_count # observation space is the number of visited_stops as well
        self.max_limit = max_limit
        self.visited_stops = [] # list of visited_stops that the agent has visited


        # Generate visited_stops
        self.Create_visited_stops()
        self.Create_Distance_Matrix()
        self.Render()

        # Initialize first point
        self.Initialise()

    # genereate the coordinates of the visited_stops
    def Create_visited_stops(self):
        coordinates = np.array(self.points)
        self.x_coordinates = coordinates[:, 0] # first column of the array being the x coordinates
        self.y_coordinates = coordinates[:, 1] # second column of the array being the y coordinates

    # generate the distance matrix between the visited_stops using the euclidean distance
    # the use of cdsit could have been used as well
    def Create_Distance_Matrix(self):
        self.stop_distances = np.zeros((self.stop_count, self.stop_count))
        for i in range(self.stop_count):
            for j in range(self.stop_count):
                self.stop_distances[i][j] = np.sqrt((self.x_coordinates[i] - self.x_coordinates[j])**2 + (self.y_coordinates[i] - self.y_coordinates[j])**2)

    # Drawing the path of the agent
    def Render(self, return_img=False):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        ax.set_title("Delivery Stops")

        # Swap x and y when plotting and invert y (which is plotted on x-axis)
        ax.scatter(self.y_coordinates, -self.x_coordinates, c="red", s=50)

        if len(self.visited_stops) > 0:
            # Swap x and y when getting coordinates and invert y
            coordinates = [self.Fetch_Coordinates(initial=True)[1], -self.Fetch_Coordinates(initial=True)[0]]
            xytext = coordinates[0] - 0.05, coordinates[1] - 0.1
            ax.annotate("START", xy=coordinates, xytext=xytext, weight="bold")

        if len(self.visited_stops) > 1:
            # Swap x and y when plotting and invert y
            ax.plot(self.y_coordinates[self.visited_stops], -self.x_coordinates[self.visited_stops], c="blue", linewidth=1, linestyle="--")
            coordinates = [self.Fetch_Coordinates(initial=False)[1], -self.Fetch_Coordinates(initial=False)[0]]
            xytext = coordinates[0] - 0.05, coordinates[1] - 0.1
            ax.annotate("END", xy=coordinates, xytext=xytext, weight="bold")

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
    
    # Initialise the environment by clearing all the visited_stops and selecting the first stop as the first stop
    def Initialise(self):
        self.visited_stops = []
        first_stop = 0
        self.visited_stops.append(first_stop)
        return first_stop

    # Take_Step function that takes the destination as an input and returns the new current_state, reward and done
    '''
    This function is called in run_episode taking as an argument the action chosen by the agent.
    '''
    def Take_Step(self, destination):
        current_state = self.Current_State()
        next_state = destination
        reward = self.Calculate_Reward(current_state, next_state)
        self.visited_stops.append(destination)
        done = len(self.visited_stops) == self.stop_count # if the number of visited_stops is equal to the number of visited_stops in the environment, then the episode is done

        return next_state, reward, done
    
    # returns the most recently visited stop's index which is the current current_state of the environment
    def Current_State(self):
        return self.visited_stops[-1]


    def Fetch_Coordinates(self,initial = False):
        current_state = self.visited_stops[0] if initial else self.Current_State()
        x = self.x_coordinates[current_state]
        y = self.y_coordinates[current_state]
        return x,y

    # calculate the reward from moving from one stop to another
    def Calculate_Reward(self, current_state, next_state):
        reward_value = self.stop_distances[current_state, next_state]
        return reward_value

def run_episode(env,agent,verbose = 1):

    # Initialise the environment
    s = env.Initialise()

    # Initialise the agent memory
    agent.Initialise_Memory()

    # Get the maximum number of visited_stops
    max_Take_Step = env.stop_count
    
    # Initialize the episode reward
    episode_reward = 0

    # added
    total_distance = 0
    
    i = 0
    while i < max_Take_Step:

        # Remember the states
        agent.Remember_State(s)

        # Choose an action
        a = agent.act(s)
        
        # Take the action, and get the reward from environment
        s_next,r,done = env.Take_Step(a)

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
        self.Initialise_Memory()

    '''
    Does take into account the current current_state but also considers the states already visited by the agent.
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
    def Initialise_Memory(self):
        self.states_memory = []

def run_n_episodes(env,agent,name="training.gif",n_episodes=2000,Render_each=100):

    # Store the rewards
    rewards = []
    imgs = []
    total_distances = []
    epsilon_values = []

    start_time = time.time()

    # Experience replay
    # the tqdm_notebook is used to show the progress bar to see our progress within the loop
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