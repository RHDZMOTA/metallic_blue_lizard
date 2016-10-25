# -*- coding: utf-8 -*-
'''

'''

import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt


# general utility functions

def vect2ind(x):
    '''
    Function to convert boolean vector to index. 
    '''
    for i in range(len(x)):
        if x[i]:
            return i

# sigmoid / tanh / linear
sigmoid = lambda x: 1 / (1 + np.exp(-x))
tanh    = lambda x: np.tanh(x)
linear  = lambda w,x: np.dot(w, x)




class adaline:
    '''
    
    '''
    desc = 'Adaline: simple neuron.'
    
    # basic information for the neuron
    def __init__(self, phi = 'linear'):
        self.phi = phi
    
    # train neuron 
    def train(x):
        return 0



class simple_logistic:
    '''
    
    '''
    desc = 'Simple logistic regression.'
    
    # basic information for the neuron
    def __init__(self, phi = 'sigmoid'):
        self.phi = phi



class multiple_logistic: 
    '''
    
    '''
    desc = 'Multiple logistic regression.'





class competitive_neurons:
    '''
    
    '''
    desc = 'Competitive Neural Network'
    
    # basic information for the neural net
    def __init__(self, neurons = 2):
        self.neurons = neurons
   


     
class feed_forward_net:
    '''
    
    '''
    desc = 'Feed forwatd propagation neural net'
    
    # basic information for the neural net
    def __init__(self, hidden_layers = 1, neurons = [1]):
        self.hidden_layers = hidden_layers
        self.neurons = neurons
        
    
    
