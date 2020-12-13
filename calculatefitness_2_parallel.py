# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 02:03:10 2020

@author: thaku
"""# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 20:56:40 2020

@author: thaku
"""

import numpy as np
import math as ma
import cv2
import copy
import timeit

from skimage.metrics import peak_signal_noise_ratio as psnr


def calculate_fitness(final_image,goal_image):
    return np.sum(np.sum(np.bitwise_xor(final_image.astype('int32'),goal_image)))

def applyRule(rule,next_state,current_state):
    current_state_padded=np.hstack((current_state,current_state[:,0:3-1]))
    current_state_padded=np.vstack((current_state_padded, current_state_padded[0:3-1,:]))
    [M, N]=current_state.shape
    pointer_row=0

    for i in range(0,M):
        pointer_column=0
        for j in range(0,N):
            patch=current_state_padded[i:i+3,j:j+3].reshape([1,9], order='F')[0]
            patch_str=''.join([str(q) for q in patch])
            patch_int=int(patch_str,2)
            ruleNo=list(rule[patch_int].values())[0]
            if(pointer_row>=M-1):
                pointer_row=-ma.floor(3/2)
            if(pointer_column>=N-1):
                pointer_column=-ma.floor(3/2)
            if(ruleNo==0):
                next_state[pointer_row+ma.floor(3/2),pointer_column+ma.floor(3/2)]=0
            elif(ruleNo==1):
                next_state[pointer_row+ma.floor(3/2),pointer_column+ma.floor(3/2)]=1
            pointer_column=pointer_column+1
        pointer_row=pointer_row+1
    return next_state


def calculateTime(func,args):
    start = timeit.default_timer()
    res=func(*args)
    stop = timeit.default_timer()
    print('Time: ', (stop - start)*1000,'ms')
    return res


def performRules(initial_state,ruleTable,passes):
    current_state=copy.deepcopy(initial_state)
    next_state=copy.deepcopy(initial_state)
    for p in range(passes):
        next_state=applyRule(ruleTable,next_state,current_state)
        current_state=copy.deepcopy(next_state)
    return current_state

def grayToBinary(start_image):
    binary_image=copy.deepcopy(start_image)
    binary_image[start_image<128]=0
    binary_image[start_image>=128]=1
    return binary_image

def binaryToGray(start_image):
    gray_image=copy.deepcopy(start_image)
    gray_image[start_image==1]=255
    return gray_image

def tempfunc(final_image,goal_image,x,passes):
    final_image=np.asarray(final_image)
    final_image_binary=grayToBinary(final_image)
    final_image_binary=performRules(final_image_binary,x,passes)
#        final_image_gray=binaryToGray(final_image_binary)
    temp=calculate_fitness(final_image_binary,goal_image)
    return temp

from joblib import Parallel, delayed
def evaluatePopulationFitness(x,mu_l,passes,final_image,goal_image):

    fitnessVals_l=Parallel(n_jobs=4)(delayed(tempfunc)(final_image,goal_image,i,passes) for i in x)

    return fitnessVals_l

def getFinalImage(start_image,bestSolution,passes):
    return binaryToGray(performRules(grayToBinary(start_image),bestSolution[0],passes))
