# -*- coding: utf-8 -*-


import copy
import numpy as np

def cross_population(individual1,individual2,f,args):
    res=f(individual1,individual2,args)
    if(np.random.uniform(0,1)<=1):
        individual1=res[0]
        individual2=res[1]
    return (individual1,individual2)
    
            
def crossoverFunction(crossfunc,args):
    matingPool=args["matingPool"]
    #crossoverRate=args["crossoverRate"]
    matingPool_array=np.array(matingPool)
    vec_cross_population=np.vectorize(cross_population,excluded=['f','args'])
    res=vec_cross_population(matingPool_array[0::2], matingPool_array[1::2],f=crossfunc,args=args)
    matingPool_array[0::2]=res[0]
    matingPool_array[1::2]=res[1]
    matingPool=matingPool_array.tolist()    
  
        
                
def crossoverStrategySelector(crossoverFunc,args):
    if(crossoverFunc=="onePointCrossover"):
        return onePointCrossover,args
    elif(crossoverFunc=="uniformCrossover"):
        return uniformCrossover,args
    elif(crossoverFunc=="nPointCrossover"):
        return nPointCrossover,args

def onePointCrossover(j,k,args):
    crossPoint=np.random.randint(0,len(j))
    temp=copy.deepcopy(j)
    j[crossPoint:]=copy.deepcopy(k[crossPoint:])
    k[crossPoint:]=copy.deepcopy(temp[crossPoint:])
    
    return [j,k]    
    
def nPointCrossover(x,y,args):
    crossoverRate=args["crossoverRate"]
    if(np.random.uniform(0,1)<crossoverRate):
        row=int(list(x.keys())[0],2)
        if(row==0):
            x1=np.random.randint(0,511,size=np.random.randint(128, 256))
            nPointCrossover.points=x1
            nPointCrossover.points.sort()
            nPointCrossover.start=False
            
        if(row in nPointCrossover.points):
            c=not nPointCrossover.start
            nPointCrossover.start=c
            
        if(nPointCrossover.start==True):
            temp=copy.deepcopy(x)
            x=copy.deepcopy(y)
            y=copy.deepcopy(temp)
    return (x,y)            
    

def uniformCrossover(x,y,args):
    crossoverRate=args["crossoverRate"]
    if(np.random.uniform(0,1)<crossoverRate):
        if(np.random.uniform(0,1)<0.5):
            temp=copy.deepcopy(x)
            x=copy.deepcopy(y)
            y=copy.deepcopy(temp)
    return (x,y)        
    
