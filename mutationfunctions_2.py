# -*- coding: utf-8 -*-

import random as ran
import copy
import math as ma
import numpy as np

def mutationFunction(mutfunc,args):
    crossedPop=args["matingPool"]
    crossedPop_array=np.array(crossedPop)
    vec_mutate_population=np.vectorize(mutfunc)
    mutatedPopulation=vec_mutate_population(crossedPop_array).tolist()
    return mutatedPopulation

def mutationStrategySelector(mutationFunc,args):
    if(mutationFunc=="uniformMutation"):
        return uniformMutation,args

def uniformMutation(rule):
    ruleNo=list(rule.values())[0]
    LB=0
    UB=1
    mutationRate=0.1
    
    res=np.random.randint(LB, UB+1)
    mutate=np.random.uniform(0,1)
    if(mutate>mutationRate):
        res=ruleNo
        
    return {list(rule.keys())[0]:res}
         