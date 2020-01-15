
# coding: utf-8

# In[1]:


import numpy as np
import pickle, gzip
np.set_printoptions(precision=3)
import math
from tqdm import trange


# In[2]:


class skynet: 
    def __init__(self, num_layers, num_neurons_layer, rate, activation, n_epoch):
        self.num_layers = num_layers
        self.num_neurons = num_neurons_layer
        self.rate = rate
        self.activation = activation # ["sigmoid", "softmax"]
        self.weights = {}
        self.new_weights = {}
        self.bias = {}
        self.layer_error = {} 
        self.layers = {}
        self.n_epoch = n_epoch
        self.act_function = {"sigmoid":self.sigmoid, "softmax":self.softmax}
    
    def oneHot(self, num):
        a = np.zeros((10,1))
        a[num] = 1
        return a

    def sigmoid(self, layer):
        new = np.ones(layer.shape, dtype = float)
        for i in range(new.shape[0]):
            new[i] = 1 / (1 + np.exp(-layer[i]))
        return new
    
    def softmax(self, layer):
        new = np.ones(layer.shape, dtype = float)
        suma = 0
        for i in range(new.shape[0]):
            new[i] = np.exp(layer[i])
            suma += new[i]
        new = new / suma
        return new
    
    def load_dataset(self):
        ''' data has to be type numpy.ndarray '''
        # iau datele dintr-un fisier cu pickle
        f = gzip.open('mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        f.close()

        # vectorii astia sunt folositi pentru antrenarea retelei
        self.train_data = train_set[0] # shape (50000,784) adica 50000 de vectori de dimensiune 784 ce reprezinta 50000 de poze cu cifre
        self.train_target = train_set[1] # shape (50000, ) reprezinta numarul desenat in fiecare imagine 

        # vecotrii astia sunt folositi pentru testarea retelei
        self.test_data = test_set[0]
        self.test_target = test_set[1]

        self.ninput = self.train_data.shape[0]
        self.ndata = self.train_data.shape[1]
    
    
    def train(self):
        self.weights[0] = np.random.normal(0.0, 1/math.sqrt(self.ndata), (self.num_neurons[0], self.ndata))
        self.bias[0] = np.ones((self.num_neurons[0],1))
        for i in range(1,self.num_layers):     
            self.weights[i] = np.random.normal(0.0, 1/math.sqrt(self.num_neurons[i-1]), (self.num_neurons[i], self.num_neurons[i-1]))      
            self.bias[i] = np.ones((self.num_neurons[i],1))
            
        self.new_weights = self.weights
       
        for ep in trange(self.n_epoch):
            for j, data in enumerate(self.train_data):
               
                # feed forward
                layer = data.reshape((data.shape[0], 1))
               
                for i in range(self.num_layers):
                    result = np.dot(self.weights[i], layer) #+ self.bias[i]
                   
                    act = self.act_function[self.activation[i]](result)
                    self.layers[i] = act
                    layer = act
                
                # backpropagation
                # last layer calc error
                last_error = self.layers[self.num_layers-1] - self.oneHot(self.train_target[j])
                #bias_error = np.dot(np.reshape(self.bias[self.num_layers-1], (1, self.num_neurons[self.num_layers-1])), error)
               
                change = np.reshape(self.layers[self.num_layers-2], (1, self.num_neurons[self.num_layers-2])) * last_error
                self.new_weights[self.num_layers-1] = self.weights[self.num_layers-1] - self.rate*change
                #self.bias[self.num_layers-1] -= bias_error*self.rate
                
                # error for remaining of layers
                for i in range(self.num_layers-2, -1, -1):
                    error = self.layers[i]*(1-self.layers[i])*np.dot(np.transpose(self.weights[i+1]), last_error)

                   
                    #bias_error = np.dot(np.reshape(self.bias[i], (1, self.num_neurons[i])), error)            
                    if(i == 0):
                       
                        change = np.reshape(data.reshape((self.ndata, 1)), (1, self.ndata)) * error
                    else:
                        change = np.reshape(self.layers[i-1], (1, self.num_neurons[i-1])) * error
                   
                    self.new_weights[i] = self.weights[i] - self.rate*change
                    last_error = error
                
               
                self.weights = self.new_weights.copy()
                #self.bias[i] -= bias_error * self.rate
                    
    def test(self):
        suma = 0
        for j, data in enumerate(self.test_data):
                layer = data.reshape((self.ndata, 1))
                for i in range(self.num_layers):
                    result = np.dot(self.weights[i], layer) #+ self.bias[i]
                    act = self.act_function[self.activation[i]](result)
                    layer = act
                    
                if(self.test_target[j] == np.argmax(layer)):
                    suma += 1
                
        print(suma/self.test_data.shape[0])


# In[3]:


net = skynet(2, [100,10], 0.1, ["sigmoid", "softmax"], 1)
net.load_dataset()
net.train()


# In[26]:


net.test()

