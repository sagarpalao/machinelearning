#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 21:24:50 2017

@author: sagar
"""

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import random
import pandas as pd
from datetime import datetime

citya = []
team = []
match = []
data = []

def v_city(x):
    if x in citya:
        return citya.index(x)
    else:
        citya.append(x)
        print(x,citya.index(x))
        return citya.index(x)

def v_team(x):
    if x in team:
        return team.index(x)
    else:
        team.append(x)
        print(x,team.index(x))
        return team.index(x)

def v_match(x):
    if x in match:
        return match.index(x)
    else:
        match.append(x)
        print(x,match.index(x))
        return match.index(x)

def loadDataSet(fileName):      
    df = pd.read_csv(fileName)
    df = df[['city','team_1','team_2','winning_team','toss_winner','type_m']]
    
    for index, row in df.iterrows():
        city = v_city(row['city'])
        team_1 = v_team(row['team_1'])
        team_2 = v_team(row['team_2'])
        toss_winner = v_team(row['toss_winner'])
        type_m = v_match(row['type_m'])
        winning_team = v_team(row['winning_team'])
        temp = [city,team_1,team_2,toss_winner,type_m,winning_team]
        data.append(temp)
    
    return data

def kMedoids(D, k, tmax=100):
    
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')
    
    M = np.arange(n)
    np.random.shuffle(M)
    M = np.sort(M[:k])
   
    Mnew = np.copy(M)

    C = {}
    for t in range(tmax):
        
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:

        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

    return M, C



dataSet = np.array(loadDataSet('Matches2.csv'))
k = 4

print('city','team_1','team_2','toss_winner','type_m','winning_team')
print("Dataset:\n" + str(dataSet))

D = pairwise_distances(dataSet, metric='euclidean')

M, C = kMedoids(D, 4)

print('medoids:')
for point_idx in M:
    print( dataSet[point_idx] )

print('')
print('clustering result:')
for label in C:
    for point_idx in C[label]:
        print('label {0}:　{1}'.format(label, dataSet[point_idx]))
