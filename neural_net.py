# -*- coding: utf-8 -*-
'''
Neural Net Package

By: Rodrigo Hernández Mota

This script comes with built-in functions commonly used in artificial
intelligence applications. 
'''

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt


# general utility functions

def vect2ind(x):
    '''
    Function to convert boolean vector to index. 
    x: input x must be of the form [0,0,1,0]
    '''
    for i in range(len(x)):
        if x[i]:
            return i
            
def add_ones(x):
    '''
    Function to adds one to the x matrix
    '''
    x    = pd.DataFrame(x)
    cols = list(x.columns.values)
    x['_bias'] = 1
    x    = x[ ['_bias'] + cols]
    return x

# sigmoid / tanh / linear
sigmoid = lambda x: 1 / (1 + np.exp(-x))
tanh    = lambda x: np.tanh(x)
linear  = lambda w,x: np.dot(w, x)




class adaline:
    '''
    
    '''
    desc = 'Adaline: simple neuron.'
    
    # basic information for the neuron
    def __init__(self, phi = 'linear', w = None,
                 x_data = None, y_data = None):
        # activation function 
        self.phi = phi
        # initial weights 
        self.w   = w
        # dataset
        self.x_data = x_data
        self.y_data = y_data
    
    # train neuron 
    def train(self):
        '''
        Train neuron.
        '''
        # x data
        x = self.x_data.copy()
        x = add_ones(x)
        x = np.matrix(x)
        # expected resutl
        y = np.matrix(self.y_data.copy())
        # calculate weights 
        w = np.linalg.inv(x.T * x) * x.T * y
        self.w = w
    
    # simulate resutls
    def evaluate(self, x = False ,retr_n = False):
        if not x:
            x = np.matrix(add_ones(self.x_data.copy()))
        else:
            x = np.matrix(add_ones(x.copy()))
        w = self.w
        self.y_estimate = x * w
        if retr_n:
            return self.y_estimate

'''
# Example:

import matplotlib.pyplot as plt

# create dataset
x_d = pd.DataFrame({'x1':np.arange(100), 'x2':np.sin(np.arange(100))})
y_d = pd.DataFrame({'y':x_d.apply(lambda x: np.random.rand(1)[0]*x[0]+5*x[1]+30*np.random.rand(1)[0],
                                  axis = 1)})                                
# initialize, train and evaluate
neural_net = adaline(x_data = x_d, y_data = y_d) 
neural_net.train()
neural_net.evaluate()

# plot results
plt.plot(np.arange(100), neural_net.y_data,'b.', np.arange(100), neural_net.y_estimate, 'r-')

# second dataset
x_d = pd.DataFrame({'x1':np.arange(50)})
y_d = pd.DataFrame({'y':x_d.apply(lambda x: x[0] * np.random.rand(1)[0], axis = 1)})

# initialize, train and evaluate
neural_net2 = adaline(x_data = x_d, y_data = y_d) 
neural_net2.train()
neural_net2.evaluate()

# plot results
plt.plot(neural_net2.x_data, neural_net2.y_data,'b.', neural_net2.x_data, neural_net2.y_estimate, 'r-')


'''

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
        
    
    
