# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 20:44:51 2020

@author: thaku
"""
import numpy as np
import json
import cv2
import random

def initialize_population(initial_image, population_size=3, bonus=False):
    new_population = []
    random_truth_table = []
    if bonus == False:
      a = np.array([[random.uniform(0.25, 1.5), 0.0],
                [0.0, random.uniform(0.25, 1.5)]])
    
      M, N = initial_image.shape
      points = np.mgrid[0:N, 0:M].reshape((2, M*N))
      new_points = np.linalg.inv(a).dot(points).round().astype(int)
      x, y = new_points.reshape((2, M, N), order='F')
      indices = x + N*y

      transformed_image = np.take(initial_image, indices, mode='wrap')
      goal_image = cv2.Canny(transformed_image, 256, 256)

      new_population.append(transformed_image.tolist())
      new_population.append(goal_image.tolist())

    elif bonus == True:
      for x_pixel in range(len(initial_image)):
        for y_pixel in range(len(initial_image[0])):
          if initial_image[x_pixel][y_pixel] < 128:
            initial_image[x_pixel][y_pixel] = 0
          elif initial_image[x_pixel][y_pixel] >= 128:
            initial_image[x_pixel][y_pixel] = 255
      
      goal_image = cv2.Canny(initial_image, len(initial_image), len(initial_image[0]))

      new_population.append(initial_image.tolist())
      new_population.append(goal_image.tolist())

    for individual in range(population_size):
      random_truth_table= list()
      for bit in range(512):
        key = format(bit, 'b')
        while len(key) != 9:
          key = "0" + key
        random_truth_table.append({key: np.random.randint(2)})
      new_population.append(random_truth_table)
    
    with open("CellularAutomata.json", 'w') as o:
      o.write(json.dumps(new_population))
      
def print_population(population_json):
  population = json.loads(population_json)

  for individual in range(len(population)):
      print(population[individual])