import numpy as np
import random
import matplotlib.pyplot as plt

## READ FILE
with open('iceWorld.txt') as f:
    content = f.readlines()
content = [x.strip() for x in content]
content
map = np.array([list(item) for item in content])
nRows = map.shape[0]  # no. rows
nCols = map.shape[1]  # no. columns

map = {(i, j): map[i][j] for i in range(nRows) for j in range(nCols)}

## PARAMETERS
# relevant parameters
nActions = 4  # there are 4 actions
nEpisodes = 2000
start = (7, 0)
goal = (7, 9)
alpha = 0.999
gamma = 0.9
epsBar = 0.9

## INITIALIZING
S = [(i, j) for i in range(nRows) for j in range(nCols)]
A = ['U', 'D', 'L', 'R']
maxSteps = 10*nRows*nCols

## USEFUL FUNCTIONS
def greedy(Q, s, map):
    # no policy choice in Holes
    if map[s] == 'H':
        return 'H'

    # for other cells: choose max_a Q(s, a)
    Qs = [Q[s, a] for a in A]
    return A[Qs.index(max(Qs))]

def epsGreedy(Q, s, epsilon, map):
    randNum = random.uniform(0, 1)
    if randNum <= epsilon:
        action = random.choice(A)
    else:
        action = greedy(Q, s, map)
    return action

def up(s):
    return (s[0] - 1, s[1])

def down(s):
    return (s[0] + 1, s[1])

def left(s):
    return (s[0], s[1] - 1)

def right(s):
    return (s[0], s[1] + 1)

def slip(coordinate):
    randNum = random.uniform(0, 1)
    if randNum > 0.9:
        return coordinate - 1
    if randNum > 0.8:
        return coordinate + 1
    return coordinate

def nextStep(s, a, map):
    # if start from the absorbing state, stay there and no rewards
    if map[s] == 'G':
        return (s, 0)

    # otherwise, try to move up, down, left, or right
    def interior(s):  # check if a cell is interior (check if hit wall or not)
        return (s[0] >= 0) and (s[1] >= 0) and (s[0] < nRows) and (s[1] < nCols)

    if a == 'U':
        nextS = up(s)
    elif a == 'D':
        nextS = down(s)
    elif a == 'L':
        nextS = left(s)
    else:
        nextS = right(s)

    # if icy surface, slip
    if map[s] == 'I':
        if (a == 'U') or (a == 'D'):
            nextS = (nextS[0], slip(nextS[1]))
        if (a == 'L') or (a == 'R'):
            nextS = (slip(nextS[0]), nextS[1])

    # if hits wall: stay there and no rewards
    if not interior(nextS):
        return s, -1

    # if gets into hole: climb out and reward = -50 (plus -1 cost of moving)
    if map[nextS] == 'H':
        return s, -51

    # if reached final goal
    if map[nextS] == 'G':
        return nextS, 99

    # otherwise: incur cost -1
    return nextS, -1

def setEpsilon(episode):
    if episode < 10:
        return epsBar
    if episode <= 1000:
        return epsBar/np.floor(episode/10)
    return 0

def sarsaLearning(map):
    # initiate: Q(s,a) = 0, and no reward yet
    Q = {(s, a): 0 for s in S for a in A}
    totalRewards = []  # track reward after each episode
    policies = []  # track optimal policies after every 100 episodes

    for E in range(nEpisodes):
        epsilon = setEpsilon(E)  # set epsilon for current episode

        #initialize for iterations
        reachedGoal = False
        step = 0
        s = start
        totalReward = 0

        # choose initial action
        a = epsGreedy(Q, s, epsilon, map)

        # explore and learn
        while (step < maxSteps) and (not reachedGoal):  # stop when exceeding max. steps or have reached absorbing state
            # move 1 step
            nextS, reward = nextStep(s, a, map)

            # choosing action in the state nextS epsilon-greedily
            nextA = epsGreedy(Q, nextS, epsilon, map)

            # update value function following learned rewards
            Q[s, a] = Q[s, a] + alpha*(reward + gamma*Q[nextS, nextA] - Q[s, a])

            # update current state and action
            s = nextS
            a = nextA

            # increase step number and check whether goal has been reached
            step += 1
            reachedGoal = s == goal
            totalReward += reward

        # save reward for this episode
        totalRewards.append(totalReward)

        # save policy if 100 episodes have passed
        if (E+1) % 100 == 0:
            pol = {s: greedy(Q, s, map) for s in S}
            policies.append(pol)
    return policies, totalRewards, Q

def write_path(map, policy, s):
    if map[s] == 'G':
        return [s]
    nextS = nextStep(s, policy[s], map)
    return [s] + write_path(map, policy, nextS[0])


def qLearning(map):
    # initiate: Q(s,a) = 0, and no reward yet
    Q = {(s, a): 0 for s in S for a in A}
    totalRewards = [] # track reward after each episode
    policies = [] # track optimal policies after every 100 episodes

    for E in range(nEpisodes):
        epsilon = setEpsilon(E)  # set epsilon for current episode

        #initialize for iterations
        reachedGoal = False
        step = 0
        s = start
        totalReward = 0

        # explore and learn
        while (step < maxSteps) and (not reachedGoal):  # stop when exceeding max. steps or have reached absorbing state
            # choose action for the current state
            a = epsGreedy(Q, s, epsilon, map)

            # move 1 step
            nextS, reward = nextStep(s, a, map)

            # in Q-learning, nextA used for updating Q is only greedy (no random)
            # this nextA may not be the action the agent ends up taking (off-policy updates)
            nextA = greedy(Q, nextS, map)

            # update value function following learned rewards
            Q[s, a] = Q[s, a] + alpha*(reward + gamma*Q[nextS, nextA] - Q[s, a])

            # update current state and action
            s = nextS

            # increase step number and check whether goal has been reached
            step += 1
            reachedGoal = s == goal
            totalReward += reward

        # save reward for this episode
        totalRewards.append(totalReward)

        # save policy if 100 episodes have passed
        if (E+1) % 100 == 0:
            pol = {s: greedy(Q, s, map) for s in S}
            policies.append(pol)

    return policies, totalRewards

## FIND OPTIMAL PATHS
optSarsaPol, sarsaRewards, Q = sarsaLearning(map)
optQPol, qRewards = qLearning(map)

## TEXT FILEs CONTAINING REWARDS PER EPISODE
out_sarsa = open('reward_sarsa.txt', 'w+')
out_qlearn = open('reward_qlearn.txt', 'w+')
out_sarsa.write(f"REWARD PER EPISODE FOR SARSA\n\n")
out_qlearn.write(f"REWARD PER EPISODE FOR Q-LEARNING\n\n")
for E in range(nEpisodes):
    out_sarsa.write(f'Reward for episode {E+1}: {sarsaRewards[E]}.\n')
    out_qlearn.write(f'Reward for episode {E+1}: {qRewards[E]}.\n')
out_sarsa.close()
out_qlearn.close()

## TEXT FILES CONTAINING POLICIES AFTER EVERY 100 EPISODES
out_sarsa = open('policies_sarsa.txt', 'w+')
out_qlearn = open('policies_qlearn.txt', 'w+')
out_sarsa.write(f"POLICIES PER 100 EPISODES FOR SARSA\n\n")
out_qlearn.write(f"POLICIES PER 100 EPISODES FOR Q-LEARNING\n\n")
episode = 0
for i in range(len(optSarsaPol)):
    episode = (i+1)*100
    out_sarsa.write(f'Episode: {episode}\n\n')
    out_qlearn.write(f'Episode: {episode}\n\n')
    sarsaPol = optSarsaPol[i]
    qPol = optQPol[i]

    for i in range(nRows):
        for j in range(nCols):
            out_sarsa.write(sarsaPol[(i, j)] + ' ')
            out_qlearn.write(qPol[(i, j)] + ' ')
        out_sarsa.write(f'\n')
        out_qlearn.write(f'\n')

    out_sarsa.write(f'\n\n')
    out_qlearn.write(f'\n\n')
out_sarsa.close()
out_qlearn.close()

## PLOT REWARDS PER EPISODE FOR TWO ALGORITHMS

nObsPlt = 2000  # how many episodes to plot (because of convergence)
fig, ax = plt.subplots()
ax.plot(sarsaRewards[:nObsPlt], color = 'tab:red', label = 'SARSA', linewidth = 0.5)
ax.plot(qRewards[:nObsPlt], color = 'tab:blue', label = 'Q-Learning', linewidth = 0.5)
ax.legend()
plt.show()
fig.savefig(f"rewards_2000.png", dpi=300)

nObsPlt = 100  # how many episodes to plot (because of convergence)
fig, ax = plt.subplots()
ax.plot(sarsaRewards[:nObsPlt], color = 'tab:red', label = 'SARSA', linewidth = 0.5)
ax.plot(qRewards[:nObsPlt], color = 'tab:blue', label = 'Q-Learning', linewidth = 0.5)
ax.legend()
plt.show()
fig.savefig(f"rewards_100.png", dpi=300)
