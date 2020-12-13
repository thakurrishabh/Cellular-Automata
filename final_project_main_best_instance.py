# -*- coding: utf-8 -*-
"""
Final project

Name: Rishabh Singh Thakur
ID: 40106439
+"""

from populationInitializer import initialize_population
from calculatefitness_2 import evaluatePopulationFitness, getFinalImage, grayToBinary
from STS_2 import solutionTaggerSorter
import statistics as stat
import copy
import numpy as np
import json
import cv2
from progressOfEvolutionHelper import progressOfEvolution,POEOfAllGAs

from parentSelectionStrategies_2 import exponentialRanking,fitnessProportionateSelectionWithoutScaling
from crossoverFunctions_2 import crossoverFunction,crossoverStrategySelector
from mutationfunctions_2 import mutationFunction,mutationStrategySelector
from survivorSelectionStrategies_2 import muPlusLambdaSelection
from diversityMeasurementStrategies_2 import calculateDiversity
from crowdingTechnique import crowding
import timeit
import os
import csv

import matplotlib.pyplot as plt

def visualize_image(image):
    plt.imshow(image,cmap="gray")
    plt.show()

def save_image(data_path,image,name,run):
    plt.imshow(image,cmap="gray")
    plt.savefig(os.path.join(data_path,name+' for Run '+str(run+1)+'.png'))

def convert(o):
    if isinstance(o, np.int64): return int(o)
    if isinstance(o, np.int32): return int(o)
    if isinstance(o, np.int16): return int(o)
    print("use the module calculatefitness_2_parallel instead of calculatefitness_2 and run again" )
    raise TypeError

def save_data(dest_file,data):
    with open(dest_file, 'w', encoding='utf-8') as output_file:
        for dic in data:
            json.dump(dic, output_file,default=convert)
            output_file.write("\n")

def save_values(dest_path,data,variable_names):
    with open(dest_path, 'w', encoding='utf-8') as output_file:
        count=0
        for dic in data:
            output_file.write(variable_names[count]+" = ")
            json.dump(dic, output_file,default=convert)
            output_file.write("\n")
            count=count+1


def calculateTime(func,args):
    start = timeit.default_timer()
    res=func(*args)
    stop = timeit.default_timer()
    print('Time : ', (stop - start)/60,'min')
    return res



def GA_Version_1(Runs,generations,parent_pop_init,mating_pop,mutationRate,crossoverRate,mutationFunc,crossoverFunc,passes):
    print("GA version 1 begins here")
    minFitnessValsOfAllRuns=[]
    averageFitnessValsOfAllRuns=[]
    for r in range(0,Runs):

        random_seed=100*(r+1)
        np.random.seed(random_seed)

        LB=0
        UB=1

        initial_image = cv2.imread("binaryhand.png", 0)

        initialize_population(initial_image, parent_pop_init, bonus=False)

        with open("CellularAutomata.json",'r') as f:
          population = json.loads(f.read());

        start_image = np.asarray(population[0])
        final_image = np.asarray(population[0])
        goal_image = np.asarray(population[1])
        visualize_image(initial_image)
        visualize_image(start_image)
        visualize_image(goal_image)

        goal_image_binary=grayToBinary(goal_image)

        population_used=copy.deepcopy(population[2:])

        s=0
        minFitnessValsPerRun=[]
        averageFitnessValsPerRun=[]

        diversity_initial=calculateDiversity(population_used)
        print("Initial diversity= "+str(diversity_initial)+" for Run: "+str(r+1))
        for epoch in range(0,generations):
            parent_pop_size=len(population_used)
            parentPopfitnessVals=evaluatePopulationFitness(population_used,parent_pop_size,passes,final_image,goal_image_binary)
            [minfitness,bestSolution]=solutionTaggerSorter(population_used, parentPopfitnessVals, parent_pop_size, 1,False)
            minfitness=minfitness[0]
            averagefitness=stat.mean(parentPopfitnessVals)
            matingPool=fitnessProportionateSelectionWithoutScaling(population_used,parentPopfitnessVals,s,parent_pop_size, mating_pop)

            muArgs={"matingPool":matingPool,"mutationRate":mutationRate,"LB":LB,"UB":UB}
            crArgs={"matingPool":matingPool,"mating_pop":mating_pop,"LB":LB,"UB":UB,"crossoverRate":crossoverRate}

            mutationStrategy,args=mutationStrategySelector(mutationFunc,muArgs)
            crossoverStrategy,args1=crossoverStrategySelector(crossoverFunc,crArgs)

            crossoverFunction(crossoverStrategy,args1)
            lamb=len(matingPool)
            offsprings=mutationFunction(mutationStrategy,args)
            offspringFitnessVals=evaluatePopulationFitness(offsprings,lamb,passes,final_image,goal_image_binary)
            newGenPop=crowding(population_used,offsprings,parentPopfitnessVals,offspringFitnessVals,parent_pop_size)


            if epoch==0:
                initFitness=minfitness

            population_used=newGenPop
            averageFitnessValsPerRun.append(averagefitness)
            minFitnessValsPerRun.append(minfitness)
            diversity_current=calculateDiversity(newGenPop)
            print("diversity of current generation= "+str(diversity_current)+" for Run: "+str(r+1))
            print("epoch="+str(epoch)+" for Run: "+str(r+1))
            print("minfitness="+str(minfitness)+" for Run: "+str(r+1))
        print("initminfitness="+str(initFitness)+" for Run: "+str(r+1))
        print("finalminfitness="+str(minfitness)+" for Run: "+str(r+1))

        visualize_image(getFinalImage(start_image,bestSolution,passes))

        minFitnessValsOfAllRuns.append(minFitnessValsPerRun)
        averageFitnessValsOfAllRuns.append(averageFitnessValsPerRun)

    stats=progressOfEvolution(Runs,generations,minFitnessValsOfAllRuns,averageFitnessValsOfAllRuns)
    print("GA version 1 ends here")
    return stats


Runs=2
generations=40
init_pop_size=40

mating_pop=40
mutationRate=0.3
crossoverRate=0.8
passes=1

mutationFunc="uniformMutation"
crossoverFunc="nPointCrossover"

versions=1

stats_1=calculateTime(GA_Version_1,(Runs,generations,init_pop_size,mating_pop,mutationRate,crossoverRate,mutationFunc,crossoverFunc,passes))

stats=[stats_1]

titles=["minfitness mean POE curve", "minfitness standard dev POE curve"]

poe_path=""

if(Runs>1):
    POEOfAllGAs(stats,generations,versions,titles,poe_path)
