import numpy
import random

from config import LEARNING_RATE
from formulas import sig, inv_sig, inv_err

curr_node_id = 0

class Layer:
    def __init__(self, num_nodes, input_vals, layer_num):
        self.num_nodes = num_nodes
        self.input_vals = input_vals
        self.layer_num = layer_num
        self.weight = [[random.random() for col in range(len(input_vals))] for row in range(num_nodes)]
        self.weight_delta = [[0 for col in range(len(input_vals))] for row in range(num_nodes)]
        self.layer_net = [0 for col in range(num_nodes)]
        self.layer_out = [0 for col in range(num_nodes)]
        self.bias = (random.random() * 2) - 1

    def eval(self):
        #evaluation part
        #Get input, compute the output of layer nodes.
        #for each node in the layer find a and z
        #a = (w^T*x)+b
        #z = sig(a)
        l=self.num_nodes
        for i in range(l):

            a=self.input_vals
            b=numpy.transpose(self.weight[i])
            s=self.bias
            k = numpy.dot(a,b)+self.bias
            z = sig(k)
            self.layer_net[i] = k
            self.layer_out[i] = z
            

    def backprop(self, other):
        #use backpropagation method to update weights
        #wnew = wold-(lr*(pd(E)/pd(w))) pd --> partial derivative
        #pd(E)/pd(w) = (pd(E)/pd(y))*(pd(y)/pd(w)) for 2nd layer
        #pd(E)/pd(w) = (pd(E)/pd(y))*(pd(y)/pd(z(i.e., z is the output of node related to w)))*(pd(z)/pd(w)) for 1st layer
        l=len(self.weight)
        for m in range(l):
            for n in range(len(self.weight[m])):
                if self.layer_num == 1:
                    gradient = other.weight_delta[0][m] * self.input_vals[n] * other.weight[0][m] * inv_sig(self.layer_out[m])
                elif self.layer_num == 2:
                    self.weight_delta[m][n] = inv_sig(self.layer_out[m]) * inv_err(self.layer_out[m], other)
                    gradient = self.weight_delta[m][n] * self.input_vals[n]
                self.weight[m][n] = self.weight[m][n] - (LEARNING_RATE * gradient)

class cfile():
    def __init__(self, name, mode = 'r'):
        #self = file.__init__(self, name, mode)
        self.fh = open(name,mode)

    def w(self, string):
        self.fh.write(str(string) + '\n')
        return None

    def close(self):
        self.fh.close()