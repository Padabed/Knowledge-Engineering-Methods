import numpy as np
import matplotlib.pyplot as plt

# Input the number of games to be played
gamesCount = input('how many games? ')
gc = int(gamesCount)
game = 1

# Determine if playing against the computer is automatic or not
auto = input('auto? y/n ')

# Define possible states: Rock, Paper, Scissors
states = ['Rock', 'Paper', 'Scissors']

# Define initial probabilities for each state
prob = np.array([[1 / 3, 1 / 3, 1 / 3],
                 [1 / 3, 1 / 3, 1 / 3],
                 [1 / 3, 1 / 3, 1 / 3]])

# Set the learning rate
alfa = 0.07

# Create arrays to store individual and sum of points
indiv_points = np.zeros(gc)
sum_points = [0] * gc

# Initialize previous choice
prev = ''

# Initialize the overall result
result = 0


# Define a function to determine the winner of a round
def fight(com, user):
    if com == user:  # Draw
        print('it is a draw (0)')
    elif (com == 'r' and user == 'p') or (com == 'p' and user == 's') or (com == 's' and user == 'r'):  # Win
        print('you won (+1)')
        indiv_points[game - 2] = 1
    elif (com == 'p' and user == 'r') or (com == 's' and user == 'p') or (com == 'r' and user == 's'):  # Lose
        print('you lose (-1)')
        indiv_points[game - 2] = -1


# Define a function to update the probabilities based on the user's choice and the current probabilities
def learn2(i, now):
    if now == 'r' and prob[i][1] < 1 - (2 * alfa) and prob[i][0] > 0 + alfa and prob[i][2] > 0 + alfa:
        prob[i][1] += alfa
        prob[i][0] -= alfa / 2
        prob[i][2] -= alfa / 2
    if now == 'p' and prob[i][2] < 1 - (2 * alfa) and prob[i][0] > 0 + alfa and prob[i][1] > 0 + alfa:
        prob[i][2] += alfa
        prob[i][0] -= alfa / 2
        prob[i][1] -= alfa / 2
    if now == 's' and prob[i][0] < 1 - (2 * alfa) and prob[i][1] > 0 + alfa and prob[i][2] > 0 + alfa:
        prob[i][0] += alfa
        prob[i][1] -= alfa / 2
        prob[i][2] -= alfa / 2


# Define a function to automatically play the computer's choice
def playAuto():
    return np.random.choice(['r', 'p', 's'], p=[0.6, 0.3, 0.1])


# Define a function to select the computer's choice based on the probabilities
def comChoice2(i):
    return np.random.choice(['r', 'p', 's'], p=prob[i])


# Define a function to get the index of a state ('r', 'p', 's')
def getIndex(x):
    if x == 'r':
        return 0
    if x == 'p':
        return 1
    if x == 's':
        return 2


# Define a function to get the name of a state based on its index
def getName(i):
    return states[int(i)]


# Start the game loop for the specified number of games
while game <= gc:
    print('\nGame number:', game)
    game = game + 1

    if auto == 'y':  # Automatic mode, computer makes the choice
        user = playAuto()
        com = comChoice2(0)
    else:  # Manual mode, user makes the choice
        user = input('choose r, p, s? ')

    if prev == '':
        print('you   vs   comp')
        print(getName(getIndex(user)), 'vs', getName(getIndex(com)))
        fight(com, user)
        prev = user
    else:
        print('prob for prev:', prev, '=', prob[getIndex(prev)])
        com = comChoice2(getIndex(prev))
        print('you   vs   comp')
        print(getName(getIndex(user)), 'vs', getName(getIndex(com)))
        fight(com, user)
        learn2(getIndex(prev), user)
        prev = user

    result = result + indiv_points[game - 2]
    sum_points[game - 2] = result
    print('your score:', result)

print('\nTHE END')
print('individual points')
print(indiv_points)
print('the course of the game')
print(sum_points)
print('====== your score:', result, '=======')

# Plot the graph for the sum of points
plt.plot(sum_points, color='green', marker='o', linewidth=1, markersize=3)
plt.ylabel('Points')
plt.xlabel('Games')
plt.show()

# Plot the graph for individual points
sts = ('win', 'draw', 'lose')

win = np.count_nonzero(indiv_points == 1)
draw = np.count_nonzero(indiv_points == 0)
lose = np.count_nonzero(indiv_points == -1)
print('Wins:', win, ', Draws:', draw, ', Loses:', lose)
count = {
    'count': (win, draw, lose)
}

plt.show()
