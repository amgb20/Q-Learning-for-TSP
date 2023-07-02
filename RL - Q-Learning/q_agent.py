#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import numpy as np 
import random

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
        self.q_table = np.zeros([states_size, actions_size]) 

        self.memory = Memory()

    # def Build_Model(self,states_size,actions_size):
    #     Q = np.zeros([states_size,actions_size])
    #     return Q

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
    def Update_Q_Table(self, s, a, r, s_next):
        self.q_table[s,a] += self.lr * (r + self.gamma * np.max(self.q_table[s_next,a]) - self.q_table[s,a]) # Bellman Equation
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    '''
    This method decides the next action to take given a state 's'. It uses the epsilon-greedy policy which means that
    the agent will either choose the action with the highest Q-value or a random action with equal probability.
    '''
    def act(self,s):
        q_values = self.q_table[s,:]
        return np.argmax(q_values) if np.random.rand() > self.epsilon else np.random.randint(self.actions_size)
    
    def Remember(self, state, action, reward, next_state, done):
        self.memory.save((state, action, reward, next_state, done))
    
    def Replay(self, batch_size):
        minibatch = random.sample(self.memory.cache, batch_size)
        for state, action, reward, next_state in minibatch:
            self.Train(state, action, reward, next_state)