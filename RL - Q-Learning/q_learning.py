#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import numpy as np 
import random

from memory import Memory
from base_agent import Agent

#gamma 0.95
#epsilon 1.0
'''
epsilon = 0.7, epsilon_min = 0.01, epsilon_decay = 0.999, gamma = 0.95, lr = 0.9
'''

class QAgent(Agent):
    def __init__(self, states_size, actions_size, total_steps = 100000, epsilon = 0.7, epsilon_min = 0.01, epsilon_decay = 0.999, gamma = 0.95, lr = 0.9):

        self.states_size = states_size
        self.actions_size = actions_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.lr = lr
        self.q_table = np.zeros([states_size, actions_size]) 

        # Initialize two Q-tables
        self.q_table1 = np.zeros([states_size, actions_size])
        self.q_table2 = np.zeros([states_size, actions_size])

        self.N_sa = np.zeros([states_size, actions_size])  # number of times we have visited a state-action pair
        self.total_steps = total_steps  # total number of steps taken
        self.steps_done = 0

        self.memory = Memory()

    # introducing the Q-Learning algorithm
    '''
    Q-Learning

    - s: current state
    - a: current action
    - r: reward
    - s_next: next state
    - lr: learning rate
    - gamma: discount factor

    This is introduced in the Q-Learning algorithm. It updates the Q-table with the new knowledge as follows the Bellman Equation:
    Q(s,a) = Q(s,a) + lr * (r + gamma * max(Q(s',a')) - Q(s,a))
    '''
    # # modified linear decay
    # def Update_Q_Table(self, s, a, r, s_next):
    #     self.q_table[s,a] += self.lr * (r + self.gamma * np.max(self.q_table[s_next,a]) - self.q_table[s,a]) # Bellman Equation
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon -= (self.epsilon - self.epsilon_min) / self.total_steps  # linear decay
    #     self.N_sa[s, a] += 1  # increment the number of times we have visited state-action pair (s, a)
    #     self.steps_done += 1  # increment steps
    
    # # original exponential decay --> best so far
    # def Update_Q_Table(self, s, a, r, s_next):
    #     self.q_table[s,a] += self.lr * (r + self.gamma * np.max(self.q_table[s_next,a]) - self.q_table[s,a]) # Bellman Equation
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay
    #     self.N_sa[s, a] += 1  # increment the number of times we have visited state-action pair (s, a)
    
    '''
    Double Q-Learning
    '''
    def Update_Double_Q_Table(self, s, a, r, s_next):
        # With 0.5 probability update the first Q-Table, 
        # otherwise update the second Q-Table
        if np.random.rand() < 0.5:
            a_next = np.argmax(self.q_table1[s_next, :])
            self.q_table1[s, a] += self.lr * (r + self.gamma * self.q_table2[s_next, a_next] - self.q_table1[s, a])
        else:
            a_next = np.argmax(self.q_table2[s_next, :])
            self.q_table2[s, a] += self.lr * (r + self.gamma * self.q_table1[s_next, a_next] - self.q_table2[s, a])
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # introducing the SARSA algorithm

    # act method for an epsilon-greedy policy
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