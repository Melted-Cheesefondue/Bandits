import numpy as np
import sys
import matplotlib.pyplot as plt
from PIL import Image
from numpy import random
import os.path
import copy
from BanditsHuman import reset, light_reset, one_run, change, total_reward, bandit_estimates


#global variables for the AI

theoretical_interventions = [(w, x, y, z) for w in range(-3,4) for x in range(-3,4) for y in range(-3,4) for z in range(-3,4)] 
    #this generates all 4-tuples with values in the -3 to 3 range and hence has 7*7*7*7=2401 elements. Each value in the 4-tuple is something the AI can adjust the corresponding bandit's output by.

interventions = [] #allowing interventions by the generic AI
grounded_interventions = [] #allowing interventions by the AI that knows that E, S, and W are interchangeable.

for a in theoretical_interventions:
    total = a[0] + a[1] + a[2] + a[3]
    if total == 0:
        interventions.append(a[:]) #the allowed interventions are required to sum to zero
        if a[1] >= a[2] and a[2] >= a[3]:
            grounded_interventions.append(a[:]) #ensures that only one set of numbers (ignoring N) are in the grounded interventions; the grounded interventions do not distinguish between E, S, and W

interventions_dictionary = {x:[0,0] for x in interventions} #this generates a [0,0]-initialised dictionary so that in the future the AI can lists the impact of various interventions
grounded_interventions_dictionary = {x:[0,0] for x in grounded_interventions} #same for the grounded AI



# the next functions concern the AI

def Ntime(): #returns how often the human has operated the north bandit (the one the AI is targetting)
    return bandit_estimates['N'][1]

def run_intervention(inter=(0,0,0,0), dic={}):#does a single human run with a specified AI intervention
    global change

    light_reset()
    change = inter
    one_run()
    if len(dic) > 0: #if passed a non-empty dictionary, updates it
        dic[inter][0]+=1
        dic[inter][1]+=Ntime()

    return total_reward() + [Ntime()] #returns the total root human reward, the total perceived human reward, and the AI reward (times the human has operated the north bandit)


def run_three_interventions(inter=(0,0,0,0), grounded_intervention=(0,0,0,0)): #the AI intervenes in three scenarios: no intervention, intervention, and grounded intervention
    global interventions_dictionary, grounded_interventions_dictionary
    reset() #resets everything to do with the human, the bandits, and the episode

    non_intervention_data = run_intervention((0,0,0,0),{})
    intervention_data = run_intervention(inter,interventions_dictionary)
    grounded_intervention_data = run_intervention(grounded_intervention,grounded_interventions_dictionary)

    return (non_intervention_data + intervention_data + grounded_intervention_data)


def best_intervention():#finds which intervention and grounded intervention the AI estimates is best
    best = (0,0,0,0)
    best_val = float('-inf')
    for inter in interventions:
        times = interventions_dictionary[inter][0]
        score = interventions_dictionary[inter][1]
        if times > 0:
            val = score/times
            if val > best_val:
                best_val = val
                best = inter
    grbest = (0,0,0,0)
    grbest_val = float('-inf')
    for grinter in grounded_interventions:
        grtimes = grounded_interventions_dictionary[grinter][0]
        grscore = grounded_interventions_dictionary[grinter][1]
        if grtimes > 0:
            val = grscore/grtimes
            if val > grbest_val:
                grbest_val = val
                grbest = grinter
    return [best, grbest]


def AI_training(filename='learning.txt'):#runs a full AI round of 2000 rounds of learning and optimisation, then writes the output to file
    global interventions_dictionary, grounded_interventions_dictionary
    interventions_dictionary = {x:[0,0] for x in interventions}
    grounded_interventions_dictionary = {x:[0,0] for x in grounded_interventions}

    with open(filename, 'w') as f:
        rand_interventions = copy.deepcopy(interventions)
        rand_grounded_interventions = copy.deepcopy(grounded_interventions)
        random.shuffle(rand_interventions)
        random.shuffle(rand_grounded_interventions)

        posi = 0
        grposi = 0

        tdone = False
        grtdone = False

        epsilon = 0.2

        for n in range(2000):
            bestest = True
            grbestest = True
        
            draw = random.rand()

            inter, grinter = best_intervention()

            if not tdone or draw < epsilon:
                inter = rand_interventions[posi]
                posi += 1
                if posi >= len(rand_interventions):
                    posi = 0
                    tdone = True
                bestest = False

            if not grtdone or draw < epsilon:
                grinter = rand_grounded_interventions[grposi]
                grposi += 1
                if grposi >= len(rand_grounded_interventions):
                    grposi = 0
                    grtdone = True
                grbestest = False

            print(run_three_interventions(inter,grinter) + list(inter) + list(grinter) + [bestest] + [grbestest],file = f)



        
