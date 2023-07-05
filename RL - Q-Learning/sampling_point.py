import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import imageio
import sys
import matplotlib.pyplot as mping

from tqdm import tqdm_notebook
from q_learning import QAgent

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
    def Render(self, return_img=False, background_image_path=None):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        ax.set_title("Sampling Points Stops")

        background_image_path = "../background.png"

        if background_image_path is not None:
            img = mping.imread(background_image_path)
            img = np.flipud(img)
            ax.imshow(img,aspect='auto', extent=[min(self.y_coordinates), max(self.y_coordinates), -min(self.x_coordinates), -max(self.x_coordinates)])

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

    def reset(self):
        return self.Initialise()

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

    # # r1 = 1/d(ij)
    # def Calculate_Reward(self, current_state, next_state):
    #     if self.stop_distances[current_state, next_state] != 0:
    #         reward_value = 1 / self.stop_distances[current_state, next_state]
    #     else:
    #         reward_value = -np.inf
    #     return reward_value

    # original
    # calculate the reward from moving from one stop to another
    def Calculate_Reward(self, current_state, next_state):
        reward_value = -self.stop_distances[current_state, next_state]
        return reward_value

# original
def run_episode(env,agent, verbose = 1):

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
    
    step_count = 0
    while step_count < max_Take_Step:

        # Remember the states
        agent.Remember_State(s)

        # Choose an action
        a = agent.act(s)
        
        # Take the action, and get the reward from environment
        next_state,reward,done = env.Take_Step(a)

        # Tweak the reward
        reward = 1 * reward
        
        if verbose: 
            print(next_state,reward,done)
        

        # # Update our knowledge in the Q-table
        # agent.Update_Q_Table(s,a,reward,next_state)

        # Update our knowledge in the Double Q-table
        agent.Update_Double_Q_Table(s,a,reward,next_state)

        # # inverse distance 
        # # Add the distance of the taken action
        # total_distance += env.stop_distances[s, a]
        
        # Update the caches
        episode_reward += reward
        s = next_state

        # original
        # Add the distance of the taken action
        total_distance -=reward
        
        # If the episode is terminated
        step_count += 1
        if done:
            break
            
    return env,agent,episode_reward, total_distance

# original
class SamplingPointQAgent(QAgent):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.Initialise_Memory()

    '''
    Does take into account the current current_state but also considers the states already visited by the agent.
    The agent avoids revisiting the states that it has already visited by setting their Q-value to -inf.
    '''
    def act(self,s):

        # # Get Q Vector
        # q = np.copy(self.q_table[s,:])

        # Get Q Vector from average of q_table1 and q_table2
        q = np.copy((self.q_table1[s,:] + self.q_table2[s,:]) / 2)

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

# original
def run_n_episodes(env,agent, name="training.gif",n_episodes=2000,Render_each=100):

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
        env,agent,episode_reward, total_distance = run_episode(env,agent, verbose = 0)
        rewards.append(episode_reward)
        total_distances.append(total_distance)
        epsilon_values.append(agent.epsilon)

        # print('total distance: ', total_distances)

        min_distance = min(total_distances)
        print("Minimum total distance over 1000 episodes: ", min_distance*100000)
        
        if i % Render_each == 0:
            print(agent.q_table)
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
    axs[0].plot(np.array(rewards)*1e5)  # Multiply rewards by e5

    # Plot total distances
    axs[1].set_title("Total distance over training")
    axs[1].plot(np.array(total_distances)*1e5)  # Multiply total_distances by e5

    # Plot epsilon values
    axs[2].set_title("Epsilon values over training")
    axs[2].plot(np.array(epsilon_values)*1e5)  # Multiply epsilon_values by e5

    plt.tight_layout()
    plt.show()

    # Save imgs as gif
    imageio.mimsave(name,imgs,duration=100)

    return env,agent