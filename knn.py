#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 18:07:39 2017

@author: sagar
"""

from numpy import *
import operator
from datetime import datetime

def createDataSet():
    dataSet = array([[1, 1],[1, 1],[1, 0],[0, 1],[0, 1]])
    labels = ['yes','yes','no','no','no']
    print("Dataset: "+ str(dataSet))
    print("\nLabels: "+ str(labels))
    return dataSet, labels
    
def classify(inX, dataSet, labels, k):
    
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    
    print("\nDistances: " + str(distances))
    
    sortedDistIndicies = distances.argsort()  
    
    print("\nSorted Distances Index: " + str(sortedDistIndicies))
    
    classCount={}          
    
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


dataset,labels = createDataSet()
    
x = [1.1,1.1]
k = 3

print("\nInput: " + str(x))
print("K: "+str(k)) 

dt = datetime.now()   
classified = classify(x,dataset,labels,k)
dt2 = datetime.now()

print("\nClassification:" + classified)

print("\nTime for evaluation: " + str(dt2.microsecond - dt.microsecond) + " microseconds")
