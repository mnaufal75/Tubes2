#!/usr/bin/env python
# coding: utf-8

# In[201]:


import numpy as np
import pandas as pd


# In[202]:


# from sklearn.datasets import load_breast_cancer

# breast_cancer = load_breast_cancer()
# X_train = breast_cancer.data
# y_train = breast_cancer.target


# In[203]:


X_train = np.asarray([[1, 2],
                      [3, 4],
                      [5, 6],
                      [7, 8],
                      [9, 10]])
y_train = np.asarray([[1],
                     [0],
                     [1],
                     [0],
                     [1]])


# In[178]:


# print(breast_cancer.data)


# In[204]:


import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Fungsi sigmoid untuk numpy array
sigmoid_v = np.vectorize(sigmoid)


# In[205]:


class Dense:
    def __init__(self, layers, activation):
        # self.layers = np.zeros((1, layers))
        # self.layers = np.random.randn(layers[l], layers[l-1])*0.01
        self.layers = layers
        self.activation = activation


# In[206]:


class Sequential:
    def __init__(self, input_shape, denses):
        self.denses = denses
        self.input_shape = input_shape
        self.weights = []
    
    def summary():
        None
    
    def set_weight(self):
        last_dense = 0
        for j, dense in enumerate(self.denses):
            if (j == 0):
                # print("Epoch: ", epoch, dense.layers)
                self.weights.append(np.random.randn(self.input_shape, dense.layers))
                self.weights.append(np.random.rand(1, dense.layers))
                last_dense = dense.layers
            else:
                # print("Epoch: ", epoch, last_dense, dense.layers)
                self.weights.append(np.random.randn(last_dense, dense.layers))
                self.weights.append(np.random.rand(1, dense.layers))
                last_dense = dense.layers
    
    def get_weight(self):
        # print(self.weights)
        return self.weights
    
    def compile():
        None
    
    def feedForward(self, X_train, y_train, mat):
        result = X_train
        it = iter(self.weights)
        for w in it:
            result = np.matmul(result, w) + next(it)
            # print(result)
            mat.append(result)

        return sigmoid_v(result)
    
    
    def backPropagation(self, X_train, y_train, target, mat):
        # print("mat", mat)
        err = []
        for i, m in enumerate(reversed(mat)):
            if i == 0:
                # print("err: ", m * (1 - m) * (target - m))
                # print("m: ", m)
                err.insert(0, m * (1 - m) * (target - m))
            else:
                # print("err: ", m * (1 - m) * self.get_weight()[len(m) - i - 1] * err[0])
                # print("m: ", m)
                err.insert(0, m * (1 - m) * err[len(err) - 1])
        return err
    
    def updateWeights(self, X_train, y_train, mat, err, momentum, learning_rate):
        print(mat)
        print(err)
        if (momentum):
            None
        else:
            it = iter(self.weights)
            for i, w in enumerate(it):
                print(i, " a", w)
                print(i, " b", w + (learning_rate * np.multiply(err[i], mat[i])))
                self.weights[i] = w + (learning_rate * np.multiply(err[i], mat[i]))
                next(it)
    
    def fit(self, X_train, y_train, epochs):
        mat = []
        self.set_weight()
        for epoch in range(epochs):
            self.feedForward(X_train, y_train, mat)
            err = self.backPropagation(X_train, y_train, 0, mat)
            self.updateWeights(X_train, y_train, mat, err, momentum=False, learning_rate=0.2)
        return mat


# In[207]:


model = Sequential(2, [
    Dense(5, 'sigmoid'),
    Dense(3, 'sigmoid'),
    Dense(1, 'sigmoid')
])

model.fit(X_train[0], y_train, 1)

