# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

def progressOfEvolution(Runs,generations,maxFitnessVals,averageFitnessVals):
    maxFitnessVals_transpose=zip(*maxFitnessVals)
    maxValsMeanOverRuns=[]
    maxValsStdevOverRuns=[]
    for val in maxFitnessVals_transpose:

        maxValsMeanOverRuns.append(np.mean(val))
        maxValsStdevOverRuns.append(np.std(val))

    averageFitnessVals_transpose=zip(*averageFitnessVals)
    avgValsMeanOverRuns=[]
    avgValsStdevOverRuns=[]
    for val in averageFitnessVals_transpose:
        avgValsMeanOverRuns.append(np.mean(val))
        avgValsStdevOverRuns.append(np.std(val))

    return [maxValsMeanOverRuns,avgValsMeanOverRuns,maxValsStdevOverRuns,avgValsStdevOverRuns]

def POEOfAllGAs(stats,generations,versions,titles,poe_path):
    labels="GA instance "
    linestyles=[":","--","-"]

    plt.figure(2)
    for i in range(0,versions):
        plt.plot(range(0,len(stats[i][0])),stats[i][0],linestyles[i],label=labels+str(i+1))
    plt.xlabel('generations')
    plt.ylabel('mean')
    plt.title(titles[0])
    plt.legend()
    plt.grid(True)
    plt.savefig(poe_path+titles[0]+'.png')
    plt.show()

        
    plt.figure(3)
    for i in range(0,versions):
        plt.plot(range(0,len(stats[i][1])),stats[i][1],linestyles[i],label=labels+str(i+1))
    plt.xlabel('generations')
    plt.ylabel('mean')
    plt.title("averagefitness mean POE curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(poe_path+"averagefitness mean POE curve"+'.png')
    plt.show()

    plt.figure(4)
    for i in range(0,versions):
        plt.plot(range(0,len(stats[i][2])),stats[i][2],linestyles[i],label=labels+str(i+1))
    plt.xlabel('generations')
    plt.ylabel('standard deviation')
    plt.title(titles[1])
    plt.legend()
    plt.grid(True)
    plt.savefig(poe_path+titles[1]+'.png')
    plt.show()

    plt.figure(5)
    for i in range(0,versions):
        plt.plot(range(0,len(stats[i][3])),stats[i][3],linestyles[i],label=labels+str(i+1))
    plt.xlabel('generations')
    plt.ylabel('standard deviation')
    plt.title("averagefitness standard dev POE curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(poe_path+"averagefitness standard dev POE curve"+'.png')
    plt.show()