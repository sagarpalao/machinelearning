#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 19:43:26 2017

@author: sagar
"""

import pandas as pd
import numpy as np

def prediction(w,x):
    result = w[0]
    for i in range(len(x)):
        result = result + w[i+1]*x[i]
    return result

df = pd.read_csv('sample_reg.csv')
train_x = df[['X']]
train_y = df[['Y']]

print(train_x)
print(train_y)

x = train_x.as_matrix()
y = train_y.as_matrix()

epochs = 50

w = [0.4,0.8]

for i in range(epochs):    
    for j in range(len(x)):        
        for k in range(len(w)):                        
            if k == 0:
                w[k] = w[k] - 0.001 * (y[j][0] - prediction(w,x[j]))
            else:
                w[k] = w[k] - 0.001 * (y[j][0] - prediction(w,x[j])) * x[j][k - 1]                
    print("Epoch " + str(i + 1) + " > " + str(w))
    
for i in range(len(x)):   
    print("Expected=%.3f, Predicted=%.3f" % (y[i][0], prediction(w,x[i])))
