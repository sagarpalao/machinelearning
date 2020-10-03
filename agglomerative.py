#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 19:28:50 2017

@author: sagar
"""
from numpy import *

class Cluster:
    def __init__(self):
        pass

    def __repr__(self):
        return '(%s,%s)' % (self.left, self.right)
  
    def add(self, clusters, grid, lefti, righti):
        self.left = clusters[lefti]
        self.right = clusters[righti]
        for r in grid:
            r[lefti] = min(r[lefti], r.pop(righti))
        grid[lefti] = map(min, zip(grid[lefti], grid.pop(righti)))
        clusters.pop(righti)
        return (clusters, grid)

def agglomerate(labels, grid):

    clusters = labels
    while len(clusters) > 1:
        print(clusters)
        print("")
        distances = [(1, 0, grid[1][0])]
        for i,row in enumerate(grid[2:]):
            distances += [(i+2, j, c) for j,c in enumerate(row[:i+2])]
        j,i,_ = min(distances, key=lambda x:x[2])
        c = Cluster()
        clusters, grid = c.add(clusters, grid, i, j)
        clusters[i] = c
    return clusters.pop()

if __name__ == '__main__':
    
    Labels = ['BA','FI','MI','NA','RM','TO']
 
    Distance = [
                  [  0, 662, 877, 255, 412, 996],
                  [662,   0, 295, 468, 268, 400],
                  [877, 295,   0, 754, 564, 138],
                  [255, 468, 754,   0, 219, 869],
                  [412, 268, 564, 219,   0, 669],
                  [996, 400, 138, 869, 669,   0]
               ]
    
    print("Dataset:")
    print(Labels)
    print(mat(Distance))
    print("")
    print("Tree\n")
    print(agglomerate(Labels, Distance))

