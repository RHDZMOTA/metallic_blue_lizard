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

def perform(y, ye):
    err = y - ye
    err = np.matrix(err)
    return np.array(err * err.T / len(y)).reshape(1,)[0]

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
    return x.values

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
        x = self.x_data.values
        x = add_ones(x)
        x = np.asmatrix(x)
        # expected resutl
        y = np.asmatrix(self.y_data.values)
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
'''
def ln_ye(ye):
    rtn = ye
    mask = (ye == 0.)
    rtn[mask] = np.float('inf')
    rtn[mask == False] = np.log(ye[mask == False])
    return ye

def ln_1_ye(ye):
    rtn = 1-ye
    mask = (1-ye) == 0.
    rtn[mask] = np.float('inf')
    rtn[mask == False] = np.log(1-ye[mask == False])
    return rtn

J_log = lambda y, ye: sum(- y * ln_ye(ye) - (1-y) * ln_1_ye(ye)) 
'''    
J_log = lambda y, ye: sum(- y * np.log(ye) - (1-y) * np.log(1-ye))
'''
def J_log(y, ye):
    jl = 0
    for i, j in zip(y, ye):
        if i == 0.:
            if j == 1.:
                jl = jl - np.float('inf')
            else:
                jl = jl - np.log(1-j)
        if i == 1.:
            if j == 0.:
                jl = jl - np.float('inf')
            else:
                jl = jl - np.log(j)
         
    return jl
'''
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
        y = y.reshape((y.shape[0], ))
        # 
        v = x * w.T
        # estimated resutls
        y_estimate = np.array(eval(self.phi+'(v)'))
        y_estimate = y_estimate.reshape((y.shape[0],))
        # error 
        error = np.matrix(y - y_estimate)
        # np.matrix([np.asscalar(x) for x in error.T])
        # calc. cost func
        j = eval(self.cost_f+'(y,y_estimate)')
        # calc. gradient (just for J)
        dj =  np.array(- error * x / x.shape[0])[0]
        return j, dj
    
    # train neuron
    def train(self):
        # import optimization package and warnings...
        from scipy.optimize import minimize
        from warnings import filterwarnings
        # weights hypothesis
        self.w = np.zeros(self.x_data.shape[1]+1)
        # optimize 
        filterwarnings('ignore')
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
            
    def mse_perform(self):
        y = self.y_data.values
        ye = self.y_estimate.values.reshape((len(y), ))
        return perform(y, ye)
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
    Competitive neural network - unsupervised learning
    '''
    desc = 'Competitive Neural Network - Unsupervised Learning'
    
    # basic information for the neural net
    def __init__(self, neurons = 2, x_data = None):
        # number of neurons 
        self.neurons = neurons
        # dataset
        self.x_data = x_data
        
    def train(self, set_weights = False, eta = 0.1, max_iter = 200,
              print_status = False):
        import numpy.matlib as npm
        # dim of the data 
        dim_data = self.x_data.shape[1]
        # initial random weights 
        if type(set_weights) == type(np.array([])):
            w = set_weights
        else:
            w = np.random.rand(dim_data, self.neurons)
            for i in range(dim_data):
                m_x = self.x_data.max(0)[i]
                m_n = self.x_data.min(0)[i]
                rng = m_x - m_n
                w[i] = w[i] * rng + m_n
        w = np.asmatrix(w)
        # data matrix 
        X = np.asmatrix(self.x_data.values)
        X = X.T
        # unsupervised training
        for i in range(max_iter):
            for k in range(self.x_data.shape[0]):
                
                # distance
                V = []
                for j in range(self.neurons):
                    D = X[:,k] - w[:,j]
                    V.append(np.asscalar(D.T.dot(D)))
                
                # closest neuron 
                ind = np.argmin(V)
                
                # localize neuron 
                y = npm.zeros((self.neurons, 1))
                y[ind, :] = 1
                
                # update weights 
                w[:,ind] = w[:, ind] + eta * (X[:, k] - w[:, ind])
        self.w_raw = w
        self.w = pd.DataFrame(w.tolist())
    
    def evaluate(self, print_results = False):
        import numpy.matlib as npm
        # data matrix 
        X = np.asmatrix(self.x_data.values)
        X = X.T
        # output layer
        y = npm.zeros((self.neurons, self.x_data.shape[0]))
        # calculate results
        w = self.w_raw
        for k in range(self.x_data.shape[0]):
            # distance
            V = []
            for j in range(self.neurons):
                D = X[:,k] - w[:,j]
                V.append(np.asscalar(D.T.dot(D)))
            
            # closest neuron
            ind = np.argmin(V)
            
            # localize neuron
            y[ind, k] = 1
        self.y_vect = y.T.tolist()
        self.y = pd.DataFrame({'y':[vect2ind(x) for x in self.y_vect]})
        ne_rons = np.unique(self.y)
        self.clusters = ne_rons
        
        # run cost function
        self.cost_function(rt_rn = False)
        
        if print_results:
            print('Neurons that found a cluster: {}'.format(ne_rons))
    
    def cost_function(self, rt_rn = True):
        
        # clusters founded
        n_urons = self.clusters
        # initialize sum vector (sum of the distance in cluster)
        group_sum = np.zeros(len(n_urons))
        # initialize element vector (number of elements in cluster)
        group_elm = np.zeros(len(n_urons))
        # save values for each group
        clusters_elements = [[] for i in range(len(n_urons))]
        clusters_elements_index = [[] for i in range(len(n_urons))]
        
        for i in range(self.x_data.shape[0]):
            # identify neurone
            neurone = np.int(self.y.values[i][0])
            # identify position 
            neurone = vect2ind(n_urons == neurone)
            # extract x value
            x = self.x_data.values[i]
            # extract w value 
            extract_w = self.w.columns[self.w.columns == neurone]
            w = np.array(self.w[extract_w].values.T.tolist()[0])
            # sum and elm
            group_sum[neurone] = group_sum[neurone] + np.linalg.norm(x - w)
            group_elm[neurone] = group_elm[neurone] + 1
            # save data 
            clusters_elements[neurone].append(x)
            # save data index
            clusters_elements_index[neurone].append(self.x_data.index[i])
        
        # save cluster's size and elements 
        self.clusters_size = np.asarray(group_elm)
        self.clusters_elements = np.asarray(clusters_elements)
        self.clusters_elements_index = np.asarray(clusters_elements_index)
        # calculate general mean (j)
        group_mean = group_sum / group_elm
        j = np.mean(group_mean)
        
        # save and return 
        self.cost = j
        if rt_rn:
            return j
        

'''
# test
from scipy.io import loadmat
m = loadmat('datos2.mat')
m = m['datos2']
data = pd.DataFrame(m.T)

cn = competitive_neurons(neurons = 10, x_data = data)
cn.train()
cn.evaluate()

print('Neurons that found a cluster: {}'.format(np.unique(cn.y)))

'''






class multilayer_perceptron:
    '''
    
    '''
    desc = 'Feed forwatd propagation neural net'
    activation_options = ['sigmoid', 'tanh']
    cost_functions = ['J_log', 'MLE']
    
    # basic information for the neural net
    def __init__(self, hidden_layers = [3], output_neurons = 1, phi = activation_options[0],
                 x_data = None, y_data = None, cost_f = cost_functions[0]):
        # number of hidden layers 
        self.hidden_layers = hidden_layers
        # number of output neurons
        self.output_neurons = output_neurons
        # activation function 
        self.phi = phi
        # error function
        self.cost_f = cost_f
        # dataset
        self.x_data = x_data
        self.y_data = y_data
    
    
    
    
