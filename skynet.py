
# coding: utf-8

# In[244]:


import numpy as np
import pickle, gzip
np.set_printoptions(precision=5, suppress = True)
import math
from tqdm.notebook import tqdm
import pandas as pd
from datetime import datetime


# In[410]:


fileName = 'polidataset.csv'
class skynet: 
    def __init__(self, num_layers, num_neurons_layer, rate, activation, n_epoch):
        self.num_layers = num_layers
        self.num_neurons = num_neurons_layer
        self.rate = rate
        self.activation = activation # ["sigmoid", "softmax" ...]
        self.weights = {}
        self.new_weights = {}
        self.bias = {}
        self.layers = {}
        
        self.layer_error = {} # for log
        self.layer_change = {} # for log
        self.n_epoch = n_epoch
        self.act_function = {"sigmoid":self.sigmoid, "softmax":self.softmax, "linear":self.linear, "relu":self.relu}
        self.dataset = np.array([])
        self.originalDataset = np.array([])
        now = datetime.now()
        self.logFile = now.strftime("%m-%d-%Y-%H-%M-%S.txt")
        open(self.logFile, "w+")

    def normalize(self, data):
        
        from sklearn import preprocessing

        #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(data)
        return x_scaled
#         normalizarea presupune ca toate valorile sa fie numerice si sa fie in [0, 1]
#         upperLimit = 1
#         return np.array(((self.dataset-self.dataset.min())/(self.dataset.max()-self.dataset.min()))*upperLimit)


    def load_dataset(self, name, normalize = True):
        self.dataset = pd.read_csv(name)
        self.originalDataset = self.dataset[:]
        # frac specifica fractiunea de randuri pe care sa le returnezi intr un sample random
        # deci frac=1 zice sa returnam toate randurile (random)
        self.dataset = self.dataset.sample(frac=1)
        
#         if(normalize):
            # normalizam datele
#             self.dataset = self.normalize()
#         else:
#             self.dataset = self.originalDataset[:]

        # impart datele in tempSet + validationSet, apoi tempSet in testSet + trainingSet
        msk = np.random.rand(len(self.dataset)) <= 0.9
        tempset = self.dataset[msk]
        testset = self.dataset[~msk]
        msk = np.random.rand(len(tempset)) <= 0.9
        trainingset = tempset[msk]
        validationset = tempset[~msk]
        
        trainingset = np.array(trainingset)
        validationset = np.array(validationset)
        testset = np.array(testset)
        l = trainingset.shape[1]
        
        self.ninput = trainingset.shape[0] # number of instances
        self.ndata = trainingset.shape[1] -1  # number of atributes -1 = minus target
        
        
        return trainingset[:,0:l-1], trainingset[:,l-1], validationset[:,0:l-1], validationset[:,l-1], testset[:,0:l-1], testset[:,l-1]

    def oneHot(self, num):
        a = np.zeros((10,1))
        a[num] = 1
        return a
    
    def linear(self, layer):
        return layer
    
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
    
    def relu(self, layer):
        new = np.zeros(layer.shape, dtype = float)
        for i in range(new.shape[0]):
            if(layer[i] > 0):
                new[i] = layer[i]
        return new
    
    def createLog(self):
            f = open(self.logFile, "a")
            for i, _ in enumerate(self.layers):
                f.write(f'layer {i}\n')
                np.savetxt(f,self.layers[i], fmt="%03.8f" )
            for i in range(self.num_layers-1, -1, -1):
                f.write(f'error {i}\n')
                np.savetxt(f,self.layer_error[i], fmt="%03.8f")
                f.write("change\n")
                np.savetxt(f,self.layer_change[i], fmt="%03.8f")
            for i in range(self.num_layers):
                f.write(f"weights {i}\n")
                np.savetxt(f,self.weights[i], fmt="%03.8f")
            f.write("\n\n")
            f.close()
    
    def load_number_dataset(self):
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
        
        return train_set, valid_set, test_set
    
#
    def train(self, train_data, train_target, gd):
        miu = 0.5
       
        self.weights[0] = np.random.normal(0.0, 1/math.sqrt(self.ndata), (self.num_neurons[0], self.ndata))
        self.bias[0] = np.zeros((self.num_neurons[0],1)) + 0.01

        for i in range(1,self.num_layers):
            self.weights[i] = np.random.normal(0.0, 1/math.sqrt(self.num_neurons[i-1]), (self.num_neurons[i], self.num_neurons[i-1]))      
            self.bias[i] = np.zeros((self.num_neurons[i],1)) + 0.01
            
        self.new_weights = self.weights.copy()
        aux_b = self.bias.copy()
        rate = self.rate
        
        for ep in tqdm(range(self.n_epoch)):
            er = 0
            for j, data in enumerate(tqdm(train_data)):
                # feed forward
                
                layer = data.reshape((data.shape[0], 1))
                for i in range(self.num_layers):
                    result = np.dot(self.weights[i], layer) + self.bias[i]
                    act = self.act_function[self.activation[i]](result)         
                    self.layers[i] = act
                    layer = act
                
                #eroare pentru ultimul layer
#                 last_error = self.layers[self.num_layers-1] - self.oneHot(train_target[j])
                last_error = self.layers[self.num_layers-1] - train_target[j]
            
                change = np.reshape(self.layers[self.num_layers-2], (1, self.num_neurons[self.num_layers-2])) * last_error
                bias_change = np.dot(np.reshape(self.bias[self.num_layers-1], (1, self.num_neurons[self.num_layers-1])), last_error)
                
                # update
                self.new_weights[self.num_layers-1] = self.weights[self.num_layers-1]  - self.rate*change
                self.bias[self.num_layers-1] -= self.rate*bias_change

                # ----- for log-----
                self.layer_change[self.num_layers-1] = change 
                self.layer_error[self.num_layers-1] = last_error
                er += last_error*last_error # pt statistica
                # -------------
                # de facut gd si pt primul layer^
                # error for remaining of layers
                for i in range(self.num_layers-2, -1, -1):
                    error = self.layers[i]*(1-self.layers[i])*np.dot(np.transpose(self.weights[i+1]), last_error) 
                    
                    #----- propagated error --------
                    if(i == 0):    
                        change = np.reshape(data.reshape((self.ndata, 1)), (1, self.ndata)) * error
                    else:
                        change = np.reshape(self.layers[i-1], (1, self.num_neurons[i-1])) * error
                    bias_change = np.dot(np.reshape(self.bias[i], (1, self.num_neurons[i])), error)
                    #--------------------------
                    
                    
                    #if(gd == 0):
                    #--------bkp classic
                    self.new_weights[i] = self.weights[i] - self.rate*change 
                    self.bias[i] -=  self.rate*bias_change
            
                    if(gd == 1):
                        if(j != 0):
                            #------bkp momentum
                            momentum = miu * last_it_weights[i]
                            momentum_b = miu * last_it_bias[i]
                            #----------------------
                            self.new_weights[i] = self.weights[i] - self.rate*change + momentum
                            self.bias[i] -=  self.rate*bias_change + momentum_b
                    elif(gd == 2):
                        if( j != 0 ):
                            #-------bkp with variable learnin rate and momentul
                            momentum = miu * last_it_weights[i]
                            momentum_b = miu * last_it_bias[i]
                            if(self.layer_error[i] > 1.04*last_it_error[i]):
                                beta = 0.7
                            else:
                                beta = 1.05
                            rate = rate*beta
                            #------------
                            self.new_weights[i] = self.weights[i] - rate*change + momentum
                            self.bias[i] -=  rate*bias_change + momentum_b
                    elif(gd == 3):
                        if( j != 0):
                            # ------------conjugate bradient bkp
                            beta_k = np.dot(self.layer_change[i],np.transpose( self.layer_change[i]-last_it_change[i] ))
                            beta_k = beta_k / np.dot(last_it_change[i], np.transpose(last_it_change[i]))
                            if(j == 1):
                                minus = -1
                            else:
                                minus = 1
                            pk[i] = -self.layer_change[i] + np.matmul(beta_k,last_pk[i])*minus
                             

    
                    last_error = error
                    
                    #------ for log -----
                    self.layer_error[i] = error
                    self.layer_change[i] = change
                    #-------
                if(j == 0):
                    aux_e = self.layer_error.copy()
                    aux_c = self.layer_change.copy()
                    aux_k = self.layer_change.copy()
                    pk = self.layer_change.copy()
                    
                    
                # salve last interation pk for bp formula
                last_pk = aux_k.copy()
                aux_k = pk
                
                # save last interation change for bp formula
                last_it_change = aux_c.copy()
                aux_c = self.layer_change.copy()
                
                # save last interation error for bp formula
                last_it_error = aux_e.copy()
                aux_e = self.layer_error.copy()
                    
                # save last iteration bias for bp formula 
                last_it_bias = aux_b.copy()
                aux_b = self.bias.copy()
                
                # save last iteration weights for bp formula 
                last_it_weights = self.weights.copy()
                self.weights = self.new_weights.copy()
                
            self.createLog()
            print("Eroare: ", er/train_data.shape[0])
                
                    
    def test(self, test_data, test_target):
        suma = 0
        for j, data in enumerate(test_data):
                layer = data.reshape((self.ndata, 1))
                for i in range(self.num_layers):
                    result = np.dot(self.weights[i], layer) + self.bias[i]
                    act = self.act_function[self.activation[i]](result)
                    layer = act
                s = abs(test_target[j] - layer)
                suma += s
#                 if(np.argmax(layer) == test_target[j]):
#                     suma += 1
                
        print(suma/test_data.shape[0])
    
    def predict(self, data):
        layer = data.reshape((self.ndata, 1))
        for i in range(self.num_layers):
            result = np.dot(self.weights[i], layer) + self.bias[i]
            act = self.act_function[self.activation[i]](result)
            layer = act
        return layer


# In[411]:


# net = skynet(4, [50, 20, 10, 1], 0.0001, ["relu", "relu", "relu", "linear"], 5) #good result with simple bkp
net = skynet(3, [10, 5, 1], 0.0001, ["relu", "relu", "linear"], 5)
train, train_tar, validation, validation_tar, test, test_tar = net.load_dataset(fileName, True)
train = net.normalize(train)
test = net.normalize(test)

# net = skynet(2, [100,10], 0.1, ["sigmoid", "sigmoid", "softmax"], 1)
# train, validation, test = net.load_number_dataset()


# In[412]:


net.train(train, train_tar, 3)
net.test(test, test_tar)

# net.train(train[0],train[1], 2)
# net.test(test[0], test[1])


# In[ ]:


# weight regularization (L1 - L2)
# gradient clipping
#  LSTM memory
# net.predict(net.oneHot(test[0][0]))

