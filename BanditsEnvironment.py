import numpy as np
import sys
import matplotlib.pyplot as plt
from PIL import Image
from numpy import random
import os.path
import copy

#global constants

orientations = ['N','E','S','W'] #the four orientations/directions: north, east, south, and west.

bandit_positions = {'W':[0,30], 'S':[30,0], 'N':[30,60], 'E':[60,30]} #position of the four bandit machines, labelled W (west), S (south), etc...
human_positions = []
for j in range(1,6): #adding the remaining positions for the grid
    human_positions.append([j*10,30])
    if j != 3: #don't add the middle position twice
        human_positions.append([30,j*10])

square_positions = human_positions[:] #the human can occupy all of these positions
square_positions = square_positions + list(bandit_positions.values()) #add the bandit positions to the squares; the human cannot occupy these positions as the bandits are already there

deltas = {'N':[0,10], 'E':[10,0], 'S':[0,-10], 'W':[-10,0]} #moving north means increasing y by 10 and leaving x alone; and so on for the other three directions

human = Image.open("./human.png")
bandit = Image.open("./bandit.png")


#global variables for environment

bandit_probs = {
    x : [random.randint(0,15),random.randint(1,5)] for x in orientations
} #the mean and standard deviation of each bandit's payout, drawn at random (mean is an integer from 0 to 15, standard deviation is an integer 1 to 5.


