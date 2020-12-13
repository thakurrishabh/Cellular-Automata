# -*- coding: utf-8 -*-
"""
Final project

Name: Rishabh Singh Thakur
ID: 40106439
+"""

from populationInitializer import initialize_population
from calculatefitness_2 import evaluatePopulationFitness, getFinalImage, grayToBinary

import copy
import numpy as np
import json
import cv2
import os

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

from sewar import msssim


def save_image(data_path,image,name):
    plt.imshow(image,cmap="gray")
    plt.savefig(os.path.join(data_path,name+'.png'))


imgs=['butterfly.png','flower.png','gates.png','simpleshapes.png']
start_imgs=[]
goal_imgs=[]
img_path="Test Image Results/"

with open("BestRuleTable.json",'r') as f:
    ruleTable = json.loads(f.read());

psnr_GA={}
msssim_GA={}
fitness_GA={}

for i in range(4):
    initial_image = cv2.imread("images resized/"+imgs[i], 0)

    initialize_population(initial_image, 1, bonus=True)

    with open("CellularAutomata.json",'r') as f:
              population = json.loads(f.read());

    start_image = np.asarray(population[0])
    goal_image = np.asarray(population[1])

    start_image_binary=grayToBinary(start_image)
    plt.imshow(start_image_binary,cmap="gray")
    plt.show()
    goal_image_binary=grayToBinary(goal_image)

    name=imgs[i][:-4]

    final_image=getFinalImage(start_image,ruleTable,1)

    psnr_GA[name]=psnr(goal_image,final_image, data_range=255)
    msssim_GA[name]=msssim(goal_image,final_image, MAX=255)
    fitness_GA[name]=evaluatePopulationFitness(ruleTable,1,1,start_image,goal_image_binary)
    print(psnr_GA)
    print(msssim_GA)
    print(fitness_GA)
    print('myimage')
    save_image(img_path,start_image,"start_"+name)
    save_image(img_path,goal_image,"goal_"+name)
    save_image(img_path,final_image,"final_"+name)

#with open('',)
