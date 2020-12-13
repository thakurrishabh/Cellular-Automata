# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 20:45:19 2020

@author: thaku
"""

import math as ma

def dist_euclid(x1,x2):
    return ma.fabs(x1-x2)

def distance(pop1,pop2,pop_size):
    dist=0
    for i,j in zip(pop1,pop2):
        dist=dist+dist_euclid(list(i.values())[0],list(j.values())[0])
    return dist/pop_size


def calculateDiversity(population):
    diversity=0
    for i in range(len(population)):
        temp=0
        for j in range(0,i-1):
            temp=temp+distance(population[i],population[j],len(population))
        diversity=diversity+ma.sqrt(temp)
    return diversity/len(population)