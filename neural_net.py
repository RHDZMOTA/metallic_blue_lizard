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
    def __init__(self, phi = 'linear',
                 x_data = None, y_data = None):
        # activation function 
        self.phi = phi
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
        if not (type(x) != type(False)):
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
neural_net = adaline(x_data = x_d, y_data = y_d) 
neural_net.train()
neural_net.evaluate()

# plot results
plt.plot(neural_net.x_data, neural_net.y_data,'b.', neural_net.x_data, neural_net.y_estimate, 'r-')
    



def f(x):
    v = np.sum(x**2)
    g = 2*x
    return v,g

x0 = np.array([10, 10, 10])
minimize(f, x0, method='bfgs', jac = True)
'''

# error functions
J_log = lambda y, ye: sum(- y * np.log(ye) - (1-y) * np.log(1-ye))
# MLE

class simple_logistic:
    '''
    
    '''
    desc = 'Simple logistic regression.'
    activation_options = ['sigmoid', 'tanh']
    cost_functions = ['J_log', 'MLE']
    
    # basic information for the neuron
    def __init__(self, phi = activation_options[0], cost_f = cost_functions[0],
                 x_data = None, y_data = None):
        # activation function 
        self.phi = phi
        # error function
        self.cost_f = cost_f
        # dataset
        self.x_data = x_data
        self.y_data = y_data
    
    
    # evaluate cost function 
    def logistic_cost(self, w):
        w = np.matrix(w)
        # transform data to matrix 
        x = np.matrix(add_ones(self.x_data.copy()))
        # expected resutls
        y = np.array(self.y_data.copy())
        # 
        v = x * w.T
        # estimated resutls
        y_estimate = np.array(eval(self.phi+'(v)'))
        # error 
        error = np.matrix(y - y_estimate)
        # calc. cost func
        j = eval(self.cost_f+'(y,y_estimate)')[0]
        # calc. gradient (just for J)
        dj =  np.array(- error.T * x / x.shape[0])[0]
        return j, dj
    
    # train neuron
    def train(self):
        # import optimization package
        from scipy.optimize import minimize
        # weights hypothesis
        self.w = np.zeros(self.x_data.shape[1]+1)
        # optimize 
        temp = lambda w: self.logistic_cost(w)
        self.w = minimize(temp, self.w, method='bfgs', jac = True).x
    
    # evaluate neuron
    def evaluate(self, x = False ,retr_n = False):
        if not type(x) != type(False):
            x = np.matrix(add_ones(self.x_data.copy()))
        else:
            x = np.matrix(add_ones(x.copy()))
        w = np.matrix(self.w)
        v = x * w.T
        self.probability = pd.DataFrame({'probability':np.array(eval(
        self.phi+'(v)')).reshape((np.shape(x)[0],))})
        self.probability['probability'] = self.probability.values.reshape((np.shape(x)[0], ))
        self.y_estimate = self.probability.apply(lambda x: 1.*(x > 0.5))
        self.y_estimate = self.y_estimate.rename(columns = {'probability':'y_estimate'})
        if retr_n:
            return self.y_estimate
'''    
# example 
x1 = 10 * np.random.rand(100)
x2 = 20 * np.random.rand(100) + 5
x_d = pd.DataFrame({'x1':x1, 'x2':x2})
mask = x_d.x2.values < 17
y_d = pd.DataFrame({'y':1. * (mask)})

neuron = simple_logistic(x_data = x_d, y_data = y_d)
neuron.train()
neuron.evaluate()

# original data

plt.plot(x_d.iloc[mask].x1,x_d.iloc[mask].x2, 'r.',x_d.iloc[mask == 0].x1,x_d.iloc[mask == 0].x2, 'b.' )
mask2 = neuron.y_estimate.values.reshape((100,))
#mask2 = np.array(mask2).reshape((100,))
plt.plot(x_d.iloc[mask2 > 0.5].x1,x_d.iloc[mask2 > 0.5].x2, 'r.',x_d.iloc[mask2 < 0.5].x1,x_d.iloc[mask2 < 0.5].x2, 'b.' )
'''


class parallel_logistic: 
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
        
    
    
