# -*- coding: utf-8 -*-

from STS_2 import solutionTaggerSorter

import copy

def muPlusLambdaSelection(parentPop,offsprings,parentPopfitnessVals,offspringFitnessVals,parent_pop_size,lamb):
    
    newParentPopfitnessVals=copy.deepcopy(parentPopfitnessVals)
    newOffspringFitnessVals=copy.deepcopy(offspringFitnessVals)
    newParentPop=copy.deepcopy(parentPop)
    newOffsprings=copy.deepcopy(offsprings)
    
    newParentPopfitnessVals.extend(newOffspringFitnessVals)
    newParentPop.extend(newOffsprings)

    [keys,newGenPopulation]=solutionTaggerSorter(newParentPop, newParentPopfitnessVals, parent_pop_size+lamb, parent_pop_size,False)
    return newGenPopulation
#lamb>mu is must
def muLambdaSelection(offsprings,offspringFitnessVals,lamb,parent_pop_size):
    [keys,newGenPopulation]=solutionTaggerSorter(offsprings, offspringFitnessVals, lamb, parent_pop_size,False)
    return newGenPopulation