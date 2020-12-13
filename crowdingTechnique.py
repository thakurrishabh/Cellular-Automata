# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 02:50:46 2020

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

def crowding(parents,offsprings,parent_fitness,offspring_fitness,pop_size):
    newPop=[]
    po=list(zip(parents,offsprings))
    for i in range(0,pop_size,2):
        p1=po[i][0]
        o1=po[i][1]
        p2=po[i+1][0]
        o2=po[i+1][1]
        p1_fit=parent_fitness[i]
        o1_fit=offspring_fitness[i]
        p2_fit=parent_fitness[i+1]
        o2_fit=offspring_fitness[i+1]
        if((distance(p1,o1,pop_size)+distance(p2,o2,pop_size))<(distance(p1,o2,pop_size)+distance(p2,o1,pop_size))):
            if(o1_fit<p1_fit):
                newPop.append(o1)
            else:
                newPop.append(p1)
            if(o2_fit<p2_fit):
                newPop.append(o2)
            else:
                newPop.append(p2)
        else:
            if(o1_fit<p2_fit):
                newPop.append(o1)
            else:
                newPop.append(p2)
            if(o2_fit<p1_fit):
                newPop.append(o2)
            else:
                newPop.append(p1)
    return newPop