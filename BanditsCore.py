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
square_positions = []
for j in range(1,6): #adding the remaining positions for the grid
    square_positions.append([j*10,30])
    if j != 3: #don't add the middle position twice
        square_positions.append([30,j*10])

human_positions = square_positions[:] #the human can occupy all of these positions
square_positions = square_positions + list(bandit_positions.values()) #add the bandit positions to the squares; the human cannot occupy these positions as the bandits are already there

deltas = {'N':[0,10], 'E':[10,0], 'S':[0,-10], 'W':[-10,0]} #moving north means increasing y by 10 and leaving x alone; and so on for the other three directions

human = Image.open("./human.png")
bandit = Image.open("./bandit.png")


#global variables for environment

bandit_probs = {
    x : [random.randint(0,15),random.randint(1,5)] for x in orientations
} #the mean and standard deviation of each bandit's payout, drawn at random (mean is an integer from 0 to 15, standard deviation is an integer 1 to 5.


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

intervention = (0,0,0,0) #default initial intervention is to do nothing



#global variables for human

round = 0
reward = [] #this tracks the human reward (including AI intervention)
root_reward = [] #this tracks the human reward (excluding AI intervention)
path = [] #this tracks the actions of the human
horizon = 100 #each episode lasts this many rounds
bandit_estimates = {x : [0,0] for x in orientations} #this generates a [0,0]-initialised dictionary so that in the future the human can lists the profit from the different bandits
human_position = [30,30] #the human starts in the middle position


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
    global intervention
    if band not in orientations:
        return 0
    root_profit = np.random.normal(bandit_probs[band][0],bandit_probs[band][1]) #returns a value sampled mean and std. of the particular bandit
    extra = intervention[orientations.index(band)]
    return [root_profit, root_profit + extra] # for manipulated human, root_profit is the actual profit, root_profit + extra is the perceived profit
    # for the bribed human, root_profit + extra is the actual and perceived profit


def move_human(direction='N'): #moves the human one square around the grid and makes them activate bandits (and updates their estimates of the bandits)
    global human_position
    if direction not in orientations: #checks the direction is N, E, S, or W
        return 0
    if human_position not in human_positions: #checks the human is at a possible human position; if not, moves them to the centre and ends the function with a zero value
        human_position[0]=30
        human_position[1]=30
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



# the next functions concern the AI

def Ntime(): #returns how often the human has operated the north bandit (the one the AI is targetting)
    return bandit_estimates['N'][1]

def run_intervention(inter=(0,0,0,0), dic={}):#does a single human run with a specified AI intervention
    global intervention

    light_reset()
    intervention = inter
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


def AI_training(filename='learning.txt'):#runs a full AI round of 5000 rounds of learning and optimisation, then writes the output to file
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



        
