#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
REINFORCEMENT LEARNING

Started on the 25/08/2017


theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""



import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import sys
import random
import time
import random
import numpy as np


import utils
from memory import Memory
from base_agent import Agent

#gamma 0.95
#epsilon 1.0

class QAgent(Agent):
    def __init__(self,states_size,actions_size,epsilon = 0.7,epsilon_min = 0.01,epsilon_decay = 0.999,gamma = 0.95,lr = 0.9):
        self.states_size = states_size
        self.actions_size = actions_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.lr = lr
        self.Q = self.Build_Model(states_size,actions_size)


    def Build_Model(self,states_size,actions_size):
        Q = np.zeros([states_size,actions_size])
        return Q

    # introducing the Q-Learning algorithm
    '''
    - s: current state
    - a: current action
    - r: reward
    - s_next: next state
    - lr: learning rate
    - gamma: discount factor

    This is introduced in the Q-Learning algorithm. It updates the Q-table with the new knowledge as follows the Bellman Equation:
    Q(s,a) = Q(s,a) + lr * (r + gamma * max(Q(s',a')) - Q(s,a))
    '''
    def Train(self,s,a,r,s_next):
        self.Q[s,a] = self.Q[s,a] + self.lr * (r + self.gamma*np.max(self.Q[s_next,a]) - self.Q[s,a]) # Bellman Equation

        # apply the epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    '''
    This method decides the next action to take given a state 's'. It uses the epsilon-greedy policy which means that
    the agent will either choose the action with the highest Q-value or a random action with equal probability.
    '''
    def act(self,s):

        q = self.Q[s,:]

        if np.random.rand() > self.epsilon: # epsilon-greedy policy
            a = np.argmax(q)
        else:
            a = np.random.randint(self.actions_size)

        return a

