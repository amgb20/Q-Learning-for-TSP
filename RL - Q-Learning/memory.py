#!/usr/bin/env python
# -*- coding: utf-8 -*- 

from collections import deque

class Memory(object):
    def __init__(self,max_memory = 2000):
        self.cache = deque(maxlen=max_memory)
    
    def save(self,args):
        self.cache.append(args)

    def empty_cache(self):
        self.__init__()


