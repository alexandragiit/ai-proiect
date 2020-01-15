#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle, gzip
np.set_printoptions(precision=3)
import math
from tqdm import tnrange


# In[2]:


# iau datele dintr-un fisier cu pickle
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
f.close()

# vectorii astia sunt folositi pentru antrenarea retelei
image_set = train_set[0] # shape (50000,784) adica 50000 de vectori de dimensiune 784 ce reprezinta 50000 de poze cu cifre
target = train_set[1] # shape (50000, ) reprezinta numarul desenat in fiecare imagine 

# vecotrii astia sunt folositi pentru testarea retelei
image_test = test_set[0]
target_test = test_set[1]

#with open("dataset.csv") as csv_file:
    #csv_reader = csv.reader(csv_file, delimiter=',')


# In[16]:


class skynet:
    def __init__(self):
        pass
    
    def loadDataset(dataset, dtype):
        ''' data has to be type numpy.ndarray '''
        pass
target.shape


# In[20]:


# acestea se numesc functii de activare
def sigmoid(layer):
    # primeste ca input "layer", un vector de dimensiune (n,) , n fiind in functie de ce dimensiune are layerul
    # ex de layer: [1,2,3,4,5,6,7,8,9,10]   (dimensiune este (n,) si nu (n,1) deoarece un vector de dimensiune (n,1) arata asa [[1],[2],[3],...,[10]])
    # returneaza tot un vector de dimensiune (n,1) ce reprezinta layer care a fost modificat conform unei operatii
    # operatia este 1 /(1 + e^-i) unde i este un element din layer
    # ex de returnare: [1/(1+e^-1), 1/(1+e^-2), 1/(1+e^-3), ... , 1/(1+e^-10) ]
    
    # new este un vectore de 1 de dimensiunea lui layer -> [1,1,1,...,1] 
    new = np.ones(layer.shape, dtype = float)
    
    #new.shape = (n,)
    print(new.shape)
    for i in range(new.shape[0]):
        new[i] = 1 / (1 + np.exp(-layer[i]))
    return new 


# In[3]:


def softmax(layer):
    # la fel ca la sigmoid ca idee, diferenta consta in operatia aplicata elementelor lui layer
    new = np.ones(layer.shape, dtype = float)
    suma = 0
    for i in range(new.shape[0]):
        new[i] = np.exp(layer[i])
        suma += new[i]
    new = new / suma
    return new 


# In[22]:


def oneHot(num):
    # transforma un numar, sper exemplu 2 intr-un vector de forma [0,0,1,0,0,0,0,0,0,0].
    a = np.zeros((10,1))
    a[num] = 1
    return a


# In[6]:

def createNetwork():
    rate = 0.1 # rata o alegi tu
    
    # initializarea weighturilor, se dau numere random
    weight1 = np.random.normal(0.0, 1/math.sqrt(784), (100,784))
    weight2 = np.random.normal(0.0, 1/math.sqrt(100), (10,100))
   
    bias1 = np.ones((100,1))
    bias2 = np.ones((10,1))

    # ----------------------- initializari aici
    weight_before2 = 0
    weight_before1 = 0
    lastLayerError_before = 0
    secondLayerError_before = 0
    g_before = 0
    pk_before = 0

    print('Choose a method (write a number between 1 and 5): ')
    choose_option = int(input())

    for j in range(3):
        for i in tnrange(50000):
            
            # inputul pentru retea, poate fi considerat ca primul strat din retea
            firstLayer = image_set[i] # un vector ce contine o imagine
            firstLayer = firstLayer.reshape((firstLayer.shape[0],1)) # reshape, nu e important
            
            # un vector de dimeniunea (100,1) ce reprezinta outputul pentru fiecare neuron de pe stratul 2
            res1 = np.reshape(np.dot( weight1, firstLayer), (100,1)) + bias1
            # aplicarea functiei de activare sigmoid pentru neuronii de pe stratul 2
            act1 = sigmoid(res1)
            
            # un vector de dimeniunea (10,1) ce reprezinta outputul pentru fiecare neuron de pe stratul 3
            res2 = np.reshape(np.dot( weight2, act1), (10,1)) + bias2
            # aplicarea functiei de activare softmax pentru neuronii de pe stratul 3
            act2 = softmax(res2)

            # eroare reprezinta diferenta dinte output-ul retelei si valorile care ar fi trebuit sa rezulte. 
            lastLayerError = (act2 - oneHot(target[i]))
            # eroare de pe layerul 2, se calculeaza printr-o formula ce depinde de eroarea anterioara
            secondLayerError = act1*(1-act1) * np.dot(np.transpose(weight2),lastLayerError)
            
            # eroare si pt bias, nu e important
            bias2Error = np.dot(np.reshape(bias2,(1,10)), lastLayerError) 
            bias1Error = np.dot(np.reshape(bias1,(1,100)), secondLayerError)
            
            # formula pentru a afla modificarea pentru fiecare weight 
            changeLast = np.reshape(act1, (1,100)) * lastLayerError
            changeSecond = np.reshape(firstLayer, (1, 784)) * secondLayerError

            bias2 = bias2 - lastLayerError * rate
            bias1 = bias1 - secondLayerError * rate

            # ----------------------- initializari pentru prima iteratie aici
            if i == 0:
                weight_before2 = weight2
                weight_before1 = weight1
                lastLayerError_before = lastLayerError
                secondLayerError_before = secondLayerError
                g_before = changeLast + 0.5 * weight2 / (weight2.shape[1] * weight2.shape[0])
                pk_before = -g_before

            miu = random.uniform(0, 1)

            # aici se face update la weight-uri

            if choose_option == 1: #gradient descent GD
                weight2 = weight2 - (changeLast + 0.5*weight2/(weight2.shape[1]*weight2.shape[0]))*rate
                weight1 = weight1 - (changeSecond + 0.5*weight1/(weight1.shape[1]*weight1.shape[0]))*rate

            if choose_option == 2: #gradient descent BP with momentum GDM
                weight2 = weight2 - (changeLast + 0.5*weight2/(weight2.shape[1]*weight2.shape[0]))*rate + miu * weight_before2
                weight1 = weight1 - (changeSecond + 0.5*weight1/(weight1.shape[1]*weight1.shape[0]))*rate + miu * weight_before1
                weight_before2 = weight2
                weight_before1 = weight1

            if choose_option == 3: #variable learning rate BP with momentum GDX
                if lastLayerError > 1.04 * lastLayerError_before and secondLayerError > 1.04 * secondLayerError_before:
                    define_beta = 0.7
                else:
                    define_beta = 1.05
                rate = define_beta * rate
                weight2 = weight2 - (changeLast + 0.5 * weight2 / (weight2.shape[1] * weight2.shape[0])) * rate + miu * weight_before2
                weight1 = weight1 - (changeSecond + 0.5 * weight1 / (weight1.shape[1] * weight1.shape[0])) * rate + miu * weight_before1
                lastLayerError_before = lastLayerError
                secondLayerError_before = secondLayerError

            if choose_option == 4: #conjugate gradient BP CGP
                g = changeLast + 0.5 * weight2 / (weight2.shape[1] * weight2.shape[0])
                delta_g_before = g - g_before
                beta = (delta_g_before * g)/(g_before * g)
                pk = - g + beta * pk_before
                weight2 = weight2 - pk * rate
                weight1 = weight1 - pk * rate

                #initializare pentru before la restul iteratiilor
                g_before = g
                pk_before = pk

            if choose_option == 5: #quasi-newton BP BFGS
                A_matrix = 1 #?
                weight2 = weight2 - (changeLast + 0.5 * weight2 / (weight2.shape[1] * weight2.shape[0])) * A_matrix
                weight1 = weight1 - (changeSecond + 0.5 * weight1 / (weight1.shape[1] * weight1.shape[0])) * A_matrix

            # asta a fost trecera prin retea a unei singure imagini, si update-ul la weighturi..se face astea pentru toate cele 50000 de imaginii
    
    # aici se testeaza acuratetea retelei pentru imaginii pe care nu a fost antrenata
    suma = 0
    for i in range(0,10000):
        firstLayer = image_test[i]
        firstLayer = firstLayer.reshape((firstLayer.shape[0],1))


        res1 = np.dot( weight1, firstLayer) + bias1
        act1 = sigmoid(res1)
        
        res2 = np.dot( weight2, act1) + bias2
        act2 = softmax(res2)
        #activarea reprezinta un vector de (10,) unde pozitia numarului cel mai mare din vector reprezinta numarul prezis de retea
        #spre ex -> [0.1, 2, 3, 0,-4, 5 ,8, 10, 100, 2.2] prezice ca imaginea avea desenata numarul 8, deoarece numarul 100 se afla pe pozitia 8 in vector
        if(target_test[i] == np.argmax(act2)):
            suma += 1
        
    print(suma/10000)
        #print(target[i], np.argmax(act2))


# In[7]:


createNetwork()


# In[ ]:




