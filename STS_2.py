# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:12:20 2020

@author: thaku
"""

def solutionTaggerSorter(population, fitnessVals, size1, size2,order):
    sortedPopulation=[]
    dictOffitnessVals = { i : fitnessVals[i] for i in range(0,size1) }
    dictOfPopulation = { i : population[i] for i in range(0,size1) }
    sortedDictOffitnessVals=sorted(dictOffitnessVals.items(), key =
             lambda kv:(kv[1], kv[0]), reverse=order)
    keysOfSortedDictOffitnessVals=[ i for i, j in sortedDictOffitnessVals ]
    itemsOfSortedDictOffitnessVals=[ j for i, j in sortedDictOffitnessVals ]

    for key in keysOfSortedDictOffitnessVals[0:size2]:
        sortedPopulation.append(dictOfPopulation.get(key))
    return itemsOfSortedDictOffitnessVals[0:size2],sortedPopulation