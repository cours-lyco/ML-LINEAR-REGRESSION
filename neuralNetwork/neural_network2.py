#!/usr/bin/env python3
from math import exp 
from random import random 


#W = np.random.rand((x_dim,y_dim))*np.sqrt(1/(ni+no))
#Why does this initialization help prevent gradient problems?

class NeuralNetwork(object):
    def __init__(self, size):
        self.n_inputs = size[0]
        self.n_hidden = size[1]
        self.n_outpous = size[2]

        self.hidden_layer =  [ {'weights': [random() for index in range(1+self.n_inputs)]} for index in range(self.n_hidden)]
        self.hidden_layer = [{'weights': [random() for index in range(
            1+self.n_hidden)]} for index in range(self.n_outpous)]


    def activation(self,weights, inputs):
        s = 0
        activation = weights[-1]
        for i in range(len(weights)-1):
            activation += weights[i] * inputs[i]
        return activation

    def transfer(self, activation):
        return 1.0/(1 + exp(-activation))
      
    def forward_propagation(Netweights, inputs):
        
        for i in range(2):
            



if __name__ == '__main__':
    nn = NeuralNetwork([2, 3, 2])
