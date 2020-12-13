# -*- coding: utf-8 -*-
"""
Final Project

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
    #raise TypeError

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


def calculateTime(st,func,args):
    start = timeit.default_timer()
    res=func(*args)
    stop = timeit.default_timer()
    print('Time'+st+': ', (stop - start)/60,'min')
    return res

def GA_Version_1(version_path,Runs,generations,parent_pop_init,mating_pop,mutationRate,mutationFunc,passes,instance):
    minFitnessValsOfAllRuns=[]
    averageFitnessValsOfAllRuns=[]
    for r in range(0,Runs):
        img_path=version_path+"GA instance "+str(instance)+"/Images/"
        initial_population_path=version_path+"GA instance "+str(instance)+"/initial_population"+" for Run "+str(r+1)+".json"
        final_population_path=version_path+"GA instance "+str(instance)+"/final_population"+" for Run "+str(r+1)+".json"
        data_path=version_path+"GA instance "+str(instance)+"/"

        random_seed=100*(r+1)
        np.random.seed(random_seed)

        LB=0
        UB=1

        initial_image = cv2.imread("binaryhand.png", 0)

        initialize_population(random_seed,initial_image, parent_pop_init, bonus=False)

        with open("CellularAutomata.json",'r') as f:
          population = json.loads(f.read());

        start_image = np.asarray(population[0])
        final_image = np.asarray(population[0])
        goal_image = np.asarray(population[1])

        save_image(img_path,initial_image,"initial_image",r)
        save_image(img_path,start_image,"start_image",r)
        save_image(img_path,goal_image,"goal_image",r)

        goal_image_binary=grayToBinary(goal_image)
        save_image(img_path,goal_image_binary,"goal_image_binary",r)

        population_used=copy.deepcopy(population[2:])
        save_data(initial_population_path,population_used)

        s=0
        minFitnessValsPerRun=[]
        averageFitnessValsPerRun=[]

        diversity_initial=calculateDiversity(population_used)
        #print("Initial diversity= "+str(diversity_initial)+" for Run: "+str(r+1))
        for epoch in range(0,generations):
            parent_pop_size=len(population_used)
            parentPopfitnessVals=evaluatePopulationFitness(population_used,parent_pop_size,passes,final_image,goal_image_binary)
            [minfitness,bestSolution]=solutionTaggerSorter(population_used, parentPopfitnessVals, parent_pop_size, 1,False)
            minfitness=minfitness[0]
            averagefitness=stat.mean(parentPopfitnessVals)
            matingPool=population_used
            muArgs={"matingPool":matingPool,"mutationRate":mutationRate,"LB":LB,"UB":UB}

            mutationStrategy,args=mutationStrategySelector(mutationFunc,muArgs)

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
#            print("diversity of current generation= "+str(diversity_current)+" for Run: "+str(r+1))
#            print("epoch="+str(epoch)+" for Run: "+str(r+1))
#            print("minfitness="+str(minfitness)+" for Run: "+str(r+1))
#            print("\n")
#        print("initminfitness="+str(initFitness)+" for Run: "+str(r+1))
#        print("finalminfitness="+str(minfitness)+" for Run: "+str(r+1))
#        print("\n")
        print(" Run: "+str(r+1))
        save_values(data_path+"data"+" for Run "+str(r+1)+".json",[diversity_initial,diversity_current,initFitness,minfitness,bestSolution],
                    ["Initial diversity","final diversity","Initial minimum fitness","final minimum fitness","Best Solution"])
        save_image(img_path,getFinalImage(start_image,bestSolution,passes),"final_image_binary",r)
        save_data(final_population_path,population_used)

        minFitnessValsOfAllRuns.append(minFitnessValsPerRun)
        averageFitnessValsOfAllRuns.append(averageFitnessValsPerRun)

    stats=progressOfEvolution(Runs,generations,minFitnessValsOfAllRuns,averageFitnessValsOfAllRuns)
    return stats

def GA_Version_2(version_path,Runs,generations,parent_pop_init,mating_pop,mutationRate,crossoverRate,mutationFunc,crossoverFunc,passes,instance):
    minFitnessValsOfAllRuns=[]
    averageFitnessValsOfAllRuns=[]
    for r in range(0,Runs):
        img_path=version_path+"GA instance "+str(instance)+"/Images/"
        initial_population_path=version_path+"GA instance "+str(instance)+"/initilal_population"+" for Run "+str(r+1)+".json"
        final_population_path=version_path+"GA instance "+str(instance)+"/final_population"+" for Run "+str(r+1)+".json"
        data_path=version_path+"GA instance "+str(instance)+"/"

        random_seed=100*(r+1)
        np.random.seed(random_seed)

        LB=0
        UB=1

        initial_image = cv2.imread("binaryhand.png", 0)

        initialize_population(random_seed,initial_image, parent_pop_init, bonus=False)

        with open("CellularAutomata.json",'r') as f:
          population = json.loads(f.read());

        start_image = np.asarray(population[0])
        final_image = np.asarray(population[0])
        goal_image = np.asarray(population[1])

        save_image(img_path,initial_image,"initial_image",r)
        save_image(img_path,start_image,"start_image",r)
        save_image(img_path,goal_image,"goal_image",r)

        goal_image_binary=grayToBinary(goal_image)
        save_image(img_path,goal_image_binary,"goal_image_binary",r)

        population_used=copy.deepcopy(population[2:])
        save_data(initial_population_path,population_used)

        s=0
        minFitnessValsPerRun=[]
        averageFitnessValsPerRun=[]

        diversity_initial=calculateDiversity(population_used)
        #print("Initial diversity= "+str(diversity_initial)+" for Run: "+str(r+1))
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
#            print("diversity of current generation= "+str(diversity_current)+" for Run: "+str(r+1))
#            print("epoch="+str(epoch)+" for Run: "+str(r+1))
#            print("minfitness="+str(minfitness)+" for Run: "+str(r+1))
#            print("\n")
#        print("initminfitness="+str(initFitness)+" for Run: "+str(r+1))
#        print("finalminfitness="+str(minfitness)+" for Run: "+str(r+1))
#        print("\n")
        print(" Run: "+str(r+1))
        save_values(data_path+"data"+" for Run "+str(r+1)+".json",[diversity_initial,diversity_current,initFitness,minfitness,bestSolution],
                    ["Initial diversity","final diversity","Initial minimum fitness","final minimum fitness","Best Solution"])
        save_image(img_path,getFinalImage(start_image,bestSolution,passes),"final_image_binary",r)
        save_data(final_population_path,population_used)

        minFitnessValsOfAllRuns.append(minFitnessValsPerRun)
        averageFitnessValsOfAllRuns.append(averageFitnessValsPerRun)

    stats=progressOfEvolution(Runs,generations,minFitnessValsOfAllRuns,averageFitnessValsOfAllRuns)
    return stats

def GA_Version_3(version_path,Runs,generations,parent_pop_init,mating_pop,mutationRate,crossoverRate,mutationFunc,crossoverFunc,passes,instance):

    minFitnessValsOfAllRuns=[]
    averageFitnessValsOfAllRuns=[]
    for r in range(0,Runs):
        img_path=version_path+"GA instance "+str(instance)+"/Images/"
        initial_population_path=version_path+"GA instance "+str(instance)+"/initilal_population"+" for Run "+str(r+1)+".json"
        final_population_path=version_path+"GA instance "+str(instance)+"/final_population"+" for Run "+str(r+1)+".json"
        data_path=version_path+"GA instance "+str(instance)+"/"

        random_seed=100*(r+1)
        np.random.seed(random_seed)

        LB=0
        UB=1

        initial_image = cv2.imread("binaryhand.png", 0)

        initialize_population(random_seed,initial_image, parent_pop_init, bonus=False)

        with open("CellularAutomata.json",'r') as f:
          population = json.loads(f.read());

        start_image = np.asarray(population[0])
        final_image = np.asarray(population[0])
        goal_image = np.asarray(population[1])

        save_image(img_path,initial_image,"initial_image",r)
        save_image(img_path,start_image,"start_image",r)
        save_image(img_path,goal_image,"goal_image",r)

        goal_image_binary=grayToBinary(goal_image)
        save_image(img_path,goal_image_binary,"goal_image_binary",r)

        population_used=copy.deepcopy(population[2:])
        save_data(initial_population_path,population_used)

        s=0
        minFitnessValsPerRun=[]
        averageFitnessValsPerRun=[]

        diversity_initial=calculateDiversity(population_used)

        #print("Initial diversity= "+str(diversity_initial)+" for Run: "+str(r+1))
        for epoch in range(0,generations):
            parent_pop_size=len(population_used)
            parentPopfitnessVals=evaluatePopulationFitness(population_used,parent_pop_size,passes,final_image,goal_image_binary)
            [minfitness,bestSolution]=solutionTaggerSorter(population_used, parentPopfitnessVals, parent_pop_size, 1,False)
            minfitness=minfitness[0]
            averagefitness=stat.mean(parentPopfitnessVals)
            matingPool=exponentialRanking(population_used,parentPopfitnessVals,s,parent_pop_size, mating_pop)

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
#            print("diversity of current generation= "+str(diversity_current)+" for Run: "+str(r+1))
#            print("epoch="+str(epoch)+" for Run: "+str(r+1))
#            print("minfitness="+str(minfitness)+" for Run: "+str(r+1))
#            print("\n")
#        print("initminfitness="+str(initFitness)+" for Run: "+str(r+1))
#        print("finalminfitness="+str(minfitness)+" for Run: "+str(r+1))
#        print("\n")
        print(" Run: "+str(r+1))
        save_values(data_path+"data"+" for Run "+str(r+1)+".json",[diversity_initial,diversity_current,initFitness,minfitness,bestSolution],
                    ["Initial diversity","final diversity","Initial minimum fitness","final minimum fitness","Best Solution"])
        save_image(img_path,getFinalImage(start_image,bestSolution,passes),"final_image_binary",r)
        save_data(final_population_path,population_used)

        minFitnessValsOfAllRuns.append(minFitnessValsPerRun)
        averageFitnessValsOfAllRuns.append(averageFitnessValsPerRun)

    stats=progressOfEvolution(Runs,generations,minFitnessValsOfAllRuns,averageFitnessValsOfAllRuns)

    return stats


passes=1
Runs=2
generations=40
init_pop_size=40
mating_pop=40

print("GA Version 1 started")
###########version 1

mutationRate=[0.1,0.3,0.5]
mutationFunc="uniformMutation"

instances=3
stats=[]
version_path="GA Version 1/"
for instance in range(instances):
    print("GA Instance "+str(instance+1)+" started")
    statis=calculateTime(" taken by GA instance "+str(instance+1),GA_Version_1,(version_path,Runs,generations,init_pop_size,mating_pop,mutationRate[instance],mutationFunc,passes,instance+1))
    stats.append(statis)
    titles=["minfitness mean POE curve", "minfitness standard dev POE curve"]

poe_path=version_path

if(Runs>1):
    POEOfAllGAs(stats,generations,instances,titles,poe_path)

print("GA Version 2 started")
###########version 2

mutationRate=[0.1,0.3,0.5]
mutationFunc="uniformMutation"

crossoverRate=0.8
crossoverFunc="nPointCrossover"

instances=3
stats=[]
version_path="GA Version 2/"
for instance in range(instances):
    print("GA Instance "+str(instance+1)+" started")
    statis=calculateTime(" taken by GA instance "+str(instance+1),GA_Version_2,(version_path,Runs,generations,init_pop_size,mating_pop,mutationRate[instance],crossoverRate,mutationFunc,crossoverFunc,passes,instance+1))
    stats.append(statis)
    titles=["minfitness mean POE curve", "minfitness standard dev POE curve"]

poe_path=version_path

if(Runs>1):
    POEOfAllGAs(stats,generations,instances,titles,poe_path)

print("GA Version 3 started")
###########version 3

mutationRate=[0.1,0.3,0.5]
mutationFunc="uniformMutation"

crossoverRate=0.8
crossoverFunc="uniformCrossover"

instances=3
stats=[]
version_path="GA Version 3/"
for instance in range(instances):
    print("GA Instance "+str(instance+1)+" started")
    statis=calculateTime(" taken by GA instance "+str(instance+1),GA_Version_3,(version_path,Runs,generations,init_pop_size,mating_pop,mutationRate[instance],crossoverRate,mutationFunc,crossoverFunc,passes,instance+1))
    stats.append(statis)
    titles=["minfitness mean POE curve", "minfitness standard dev POE curve"]

poe_path=version_path

if(Runs>1):
    POEOfAllGAs(stats,generations,instances,titles,poe_path)
