# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 01:38:30 2020

@author: thaku
"""

import random as ran
import numpy as np
def stochasticUniversalSamplingAlgorithm(cdf, numberOfParentsForMating, population):
    matingPool=[0 for i in range(0,numberOfParentsForMating)]
    currentMember=1
    i=1
    r=np.random.uniform(0,1/numberOfParentsForMating)
    while(currentMember<=numberOfParentsForMating):
        while(r<=cdf[i-1]):
            matingPool[currentMember-1]=population[i-1]
            r=r+(1/numberOfParentsForMating)
            currentMember=currentMember+1
        i=i+1
    return matingPool