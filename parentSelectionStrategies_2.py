# -*- coding: utf-8 -*-

from STS_2 import solutionTaggerSorter
from samplingMethods_2 import stochasticUniversalSamplingAlgorithm
import math as ma
import statistics as stat

def parentSelectionDecorator(rankfunc):
    def SelectionFunction(*args,**kwargs):
        [sortedFitnessValues,populationSorted]=solutionTaggerSorter(args[0],args[1],args[3],args[3],False)
        p=rankfunc(*args,**kwargs)

        cdf=[]
        sum=0
        for i in p:
            sum=sum+i
            cdf.append(sum)
        return stochasticUniversalSamplingAlgorithm(cdf,args[4], populationSorted)
    return SelectionFunction

@parentSelectionDecorator
def fitnessProportionateSelectionWithoutScaling(parentPop,parentPopfitnessVals,s,parent_pop_size, mating_pop):
    p=[]
    f_sum=ma.fsum(parentPopfitnessVals)
    for i in parentPopfitnessVals:
        p.append(i/f_sum)
    return p

@parentSelectionDecorator
def fitnessProportionateSelection(parentPop,parentPopfitnessVals,s,parent_pop_size, mating_pop):
    p=[]
    f_scaled=[]
    f_mean=stat.mean(parentPopfitnessVals)
    f_std=stat.stdev(parentPopfitnessVals)
    c=2
    for f in parentPopfitnessVals:
        f_scaled.append(max((f-(f_mean-(c*f_std))),0))
    f_scaled_sum=ma.fsum(f_scaled)
    for i in parentPopfitnessVals:
        p.append(i/f_scaled_sum)
    return p

@parentSelectionDecorator
def linearRanking(parentPop,parentPopfitnessVals,s,parent_pop_size, mating_pop):
    p=[]
    for i in range(0,parent_pop_size):
        p.append(((2-s)/(parent_pop_size))+((2*i*(s-1))/(parent_pop_size*(parent_pop_size-1))))
    return p

@parentSelectionDecorator
def exponentialRanking(parentPop,parentPopfitnessVals,s,parent_pop_size, mating_pop):
    p=[]
    c=(parent_pop_size-((ma.exp(-parent_pop_size)-1)/(ma.exp(-1)-1)))
    for i in range(0,parent_pop_size):
        p.append((1-ma.exp(-i))/(c))
    return p