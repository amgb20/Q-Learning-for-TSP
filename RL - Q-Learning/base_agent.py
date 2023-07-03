#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import sys
import random
import time
import random
import numpy as np

class Agent(object):
    def __init__(self):
        pass

    '''
    not used in the code but it is common practice in ML to have a state vector capable of handling multiple states and therefore dimensions (1D to 3D).
    Useful when there are more complex environments and agents.
    '''
    def expand_state_vector(self,state):
        if len(state.shape) == 1 or len(state.shape)==3:
            return np.expand_dims(state,axis = 0)
        else:
            return state

    def remember(self,*args):
        self.memory.save(args)