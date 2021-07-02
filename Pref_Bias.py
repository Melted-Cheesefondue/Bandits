
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects
import sys
import random


class SeriesPeople:
    def __init__(self,initial_val=None):
        random.seed()
        fsize=5
        if initial_val==None:
            self.pref_bias = [[random.random() for i in range(fsize)] for j in range(2)]
        else:
            self.pref_bias = [[initial_val for i in range(fsize)] for j in range(2)]

    def out(self):
        print(self.pref_bias)

    def dist(self, ser):
        return self.gendist(ser,2)

    def sqdist(self, ser):
        return self.gendist(ser,2)


    def gendist(self, ser, pow):
        length = len(self.pref_bias)
        num = 0
        diffs = [0 for i in range(length)]
        for i in range(length):
            num += len(self.pref_bias[i])
            for pr in zip(self.pref_bias[i], ser.pref_bias[i]):
                diffs[i] += abs(pr[0]-pr[1])**pow
        total_diff = 0
        for td in diffs:
            total_diff += td
        return (td/num)**(1/pow)



def watch_length(per,ser):
    prob_next = max(1-per.sqdist(ser),0)
    goal = 5
    for i in range(goal):
        if random.random()>prob_next:
            return i
    return goal




per = SeriesPeople()

#series = [SeriesPeople() for i in range(20)]

#for ser in series:
#    print(watch_length(per,ser))

past_record=[]


def best_guess(past_try,per):
    max_len=0
    for rec in past_try:
        if rec[0]>max_len:
            max_len=rec[0]
    num=0
    for rec in past_try:
        if rec[0]==max_len:
            num+=1

    print(max_len,num)
    guess = SeriesPeople(0)
    guess.out()
    for rec in past_try:
        if rec[0]==max_len:
            for i in range(len(guess.pref_bias)):
                for j in range(len(guess.pref_bias[i])):
                    guess.pref_bias[i][j] += rec[1][i][j]/num
    guess.out()
    per.out()
    zero = SeriesPeople(0)
    one = SeriesPeople(1)
    half = SeriesPeople(0.5)
    print(per.sqdist(guess))
    print(per.sqdist(zero))
    print(per.sqdist(one))
    print(per.sqdist(half))


for i in range(20):
    ser = SeriesPeople()
    leng = watch_length(per,ser)
    past_record.append([leng,ser.pref_bias])

best_guess(past_record,per)


