#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 19:43:26 2017

@author: sagar
"""

import pandas as pd
import numpy as np
import math

def prediction(w,x):
    result = w[0]
    for i in range(len(x)):
        result = result + w[i+1]*x[i]
    result = 1.0 / (1.0 + math.exp(-result))
    return result

df = pd.read_csv('sample_log.csv')
train_x = df[['X1','X2']]
train_y = df[['Y']]

print(train_x)
print(train_y)

x = train_x.as_matrix()
y = train_y.as_matrix()

epochs = 100

w = [-0.406605464, 0.852573316, -1.104746259]

for i in range(epochs):    
    for j in range(len(x)):        
        for k in range(len(w)):  
            
            yhat = prediction(w,x[j])  
            #print(yhat)                
            if k == 0:
                w[k] = w[k] - 0.001 * (y[j][0] - yhat) * yhat * (1 - yhat)
            else:
                w[k] = w[k] - 0.001 * (y[j][0] - yhat) * yhat * (1 - yhat) * x[j][k - 1]                
    print("Epoch " + str(i + 1) + " > " + str(w))
    
for i in range(len(x)):   
    print("Expected=%.3f, Predicted=%.3f" % (y[i][0], prediction(w,x[i])))
