import numpy as np
import sys
import matplotlib.pyplot as plt
from PIL import Image
from numpy import random
import os.path
import copy
from BanditsEnvironment import orientations, bandit_positions, square_positions, human, bandit, bandit_probs, human_positions, deltas


#global variables for human

round = 0
reward = [] #this tracks the human reward (including AI intervention)
root_reward = [] #this tracks the human reward (excluding AI intervention)
path = [] #this tracks the actions of the human
horizon = 100 #each episode lasts this many rounds
bandit_estimates = {x : [0,0] for x in orientations} #this generates a [0,0]-initialised dictionary so that in the future the human can lists the profit from the different bandits
human_position = [30,30] #the human starts in the middle position

change = (0,0,0,0) #the default change to the profit of the bandits is zero; this is where the AI can intervene


# the next functions reset the variables for the human and the episode

def reset(): #resets the human variables and the episode (including the mean and std. of the bandits)
    light_reset()
    global bandit_probs
    bandit_probs = {
        x : [random.randint(0,15),random.randint(1,5)] for x in orientations
    }

def light_reset(): #resets the human variables and the episode, but not the mean and std. of the bandits
    global path, round, reward, horizon, bandit_estimates, human_position, root_reward
    path = []
    round = 0
    reward = []
    root_reward = []
    horizon = 100
    bandit_estimates = {x : [0,0] for x in orientations}
    human_position = [30,30]
    change = (0,0,0,0)




# the next functions concern the human moving around, playing the bandits, and updating their estimates

def best_bandit(): #returns the best bandit, according to human estimate
    global bandit_estimates
    rand_orient = orientations[:]
    random.shuffle(rand_orient) #this ensures that if multiple bandits are co-equal best, one is chosen at random
    best = 'F'
    best_val = float('-inf')
    for direction in rand_orient:
        est = estimate_of(direction)
        if est > best_val:
            best_val = est
            best = direction
    return best

def estimate_of(band): #returns the human estimate of the return of a given bandit
    global bandit_estimates
    if bandit_estimates[band][1] == 0:
        return 15 #if the human has not tried the bandit yet, estimates it to be 15 (this encourages exploration)
    return bandit_estimates[band][0] / bandit_estimates[band][1] #returns average reward of that bandit


def update_estimate(band,profit): #updates the human's bandit estimates after it's seen a profit
    global bandit_estimates
    bandit_estimates[band][1] += 1 #increases the number of times it has interacted with that bandit
    bandit_estimates[band][0] += profit #increases total profit/reward sum
    

def total_reward(): #returns the human's total root reward (excluding AI intervention) and total reward (including AI intervention)
    global reward, root_reward
    root_sum = 0
    sum = 0
    for k in root_reward:
        root_sum += k
    for j in reward:
        sum += j
    return [root_sum, sum]

def run_bandit(band='N'): #runs a bandit to get its reward
    global change
    if band not in orientations:
        return 0
    root_profit = np.random.normal(bandit_probs[band][0],bandit_probs[band][1]) #returns a value sampled mean and std. of the particular bandit
    extra = change[orientations.index(band)]
    return [root_profit, root_profit + extra] # for manipulated human, root_profit is the actual profit, root_profit + extra is the perceived profit
    # for the bribed human, root_profit + extra is the actual and perceived profit


#functions for moving the human around

def move_human(direction='N'): #moves the human one square around the grid and makes them activate bandits (and updates their estimates of the bandits)
    global human_position
    if human_position not in human_positions: #checks the human is at a possible human position; if not, moves them to the centre and ends the function with a zero value
        human_position[0]=30
        human_position[1]=30
        return 0
    if direction not in orientations: #checks the direction is N, E, S, or W
        return 0

    delta = deltas[direction] #translates the direction into a grid coordinate change
    new_pos = [sum(x) for x in zip(human_position,delta)] #this is the human's new position: sum of the old position and the delta

    if new_pos not in square_positions: # if the new position is not in the set of possible positions, exit the function with a 0 value
        return 0


    global round, path, reward, root_reward
    round += 1
    if round <= horizon:
        path.append(direction) #if the episode is not yet ended, adds the new action of the human to the path list
    else:
        return 0

    if new_pos in bandit_positions.values(): #the human has tried to "move" onto a bandit: this will activate that bandit, and the human will stay put
        root_profit, perceived_profit = run_bandit(direction) #gets the profits of the bandit
        root_reward.append(root_profit)
        reward.append(perceived_profit)
        update_estimate(direction, perceived_profit)
        return 2


    reward.append(0) #if the human moves, they don't activate a bandit, and get no reward
    root_reward.append(0)


    human_position[0] = new_pos[0] #if the human moves, this updates their position
    human_position[1] = new_pos[1]

    return 1

def move_human_to(destination='S'): #moves a human to just next to the bandit in the 'destination' variable
    if destination not in orientations:
        return 0

    if current_position() == destination: #if the human is already there, no move
        return 1

    ortho_delta = orthogonal_move(destination) #tells the direction the robot has to move to be on the correct axis to go to "destination"
    while ortho_delta != 'F': #moves until it's on the right axis (E-W or N-S)
        if move_human(ortho_delta) == 0: #stop moving if you reach the end of the episode or the movement fails for some other reason
            ortho_delta =  'F'
        else:
            ortho_delta = orthogonal_move(destination)

    to_move = (current_position() != destination)
    while to_move: #moves to destination (once on the correct axis)
        if move_human(destination) == 0: #stop moving if you reach the end of the episode or the movement fails for some other reason
            to_move = False
        else:
            to_move = (current_position() != destination)

    return 2

def current_position(): #checks if the human is just next to a bandit, and if so, which one
    global human_position

    for ori in orientations:
        new_pos = [sum(x) for x in zip(human_position,deltas[ori])] #checks whether there is any bandit within one square of the human
        if bandit_positions[ori] == new_pos:
            return ori
    return 'F' #if not next to a bandit, returns 'F'

def orthogonal_move(objective='S'): #finds if the human is on the right axis (N-S or E-W), and, if not, which direction it needs to move to get on it
    if objective in ['N','S']:
        if human_position[0] < 30:
            return 'E'
        if human_position[0] > 30:
            return 'W'
    if objective in ['E','W']:
        if human_position[1] < 30:
            return 'N'
        if human_position[1] > 30:
            return 'S'
    return 'F'



# this function determines what the human will do

def one_run(): #program for what humans will do in one episode
    global human_position
    indir = random.choice(orientations) #chooses a random direction to start walking in

    while round<3: #walks to the bandit in a random direction (two moves) then operates that bandit (one move)
        move_human(indir)

    while round<horizon:

        if estimate_of(current_position()) >= 10:
            target_bandit = current_position() #if the human estimates it will get 10 or more, it will stay put
        else:
            target_bandit = best_bandit() #otherwise, go to the bandit the human estimates is best
            move_human_to(target_bandit)

        move_human(target_bandit) #then, the human operates the bandit it targerts




#the next functions are graphic functions, if the various situations need to be graphed

def show_human(pl): #shows an image of the human
    global human_position, im
    im.remove()
    im = pl.imshow(human,extent=[human_position[0],human_position[0]+10,human_position[1],human_position[1]+10],zorder=15,aspect=1)

def square(pl, x, y): #places an image of a square with the bottom left corner in the given position
    rectangle = pl.Rectangle((x,y), 10, 10, fc='grey',ec="black")
    pl.gca().add_patch(rectangle)

def setup_grid(pl): #sets up the gridworld with the bandits and the initial position of the human
    ax=pl.axes()
    ax.set_aspect('equal')
    ax.set_axis_off()
    pl.text(34.4,4,'S',horizontalalignment='center',verticalalignment='center',color='red',zorder=30,fontsize=5,weight='bold') #shows the text on the bandit
    pl.text(34.4,64,'N',horizontalalignment='center',verticalalignment='center',color='red',zorder=30,fontsize=5,weight='bold')
    pl.text(4.4,34,'W',horizontalalignment='center',verticalalignment='center',color='red',zorder=30,fontsize=5,weight='bold')
    pl.text(64.4,34,'E',horizontalalignment='center',verticalalignment='center',color='red',zorder=30,fontsize=5,weight='bold')
    pl.imshow(bandit,extent=[31,39,1,9],zorder=15,aspect=1) #shows the bandit
    pl.imshow(bandit,extent=[1,9,31,39],zorder=15,aspect=1)
    pl.imshow(bandit,extent=[31,39,61,69],zorder=15,aspect=1)
    pl.imshow(bandit,extent=[61,69,31,39],zorder=15,aspect=1)
    for pos in square_positions:
        square(pl, *pos) #plots the possible squares in the grid
    global im
    im = pl.imshow(human,extent=[human_position[0],human_position[0]+10,human_position[1],human_position[1]+10],zorder=15,aspect=1) #shows the position of the human
    pl.axis('scaled')

def control(event, fig, go_list): #changes the colour of the background if an impossible manoever is tried when manually moving the robot
    key_pressed = event.key
    if key_pressed is not None and key_pressed != 'shift':
        if key_pressed.upper()=='Q':
            go_list[0] = False
        else:
            val = move_human(key_pressed)
            if val==0:
                fig.set_facecolor((1.0, 0.47, 0.42)) #colours the background red if an impossible maneover is attempted
            else:
                fig.set_facecolor((1.0, 1.0, 1.0))
                show_human(plt)

def bandit_GUI(): #runs the bandit gui
    fig, ax = plt.subplots()
    setup_grid(plt)
    show_human(plt)

    plt.draw()
    go_list = [True]
    while go_list[0]==True:
        cid = fig.canvas.mpl_connect('key_press_event', lambda event: control(event,fig,go_list))
        plt.waitforbuttonpress(0)
        fig.canvas.mpl_disconnect(cid)

    plt.pause(0.5)
    plt.close()




