import numpy as np

import json
from sklearn.utils import resample

# make train/test dividing reproducible for debugging
np.random.seed(0)

#####################################################
#
#   For the original dataset found here: https://github.com/mshr-h/moonboard-ml
#
##########################################################

train_filename = "train.json"
test_filename = "test.json"

def get_problem_set(filename):

    rows = 18
    cols = 11

    # read in the JSON file and extract the "Grade" values
    grades = []
    problems = []
    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            grades.append(data['Grade'])
            problems.append(data['Holds'])

    holds = []
    for sublist in problems:
        board = [[0 for i in range(cols)] for j in range(rows)]
        for data in sublist:
            row = int(data[1:]) - 1
            col = ord(data[0]) - ord('A')
            board[row][col] = 1

        holds.append(board)

    return(np.array(holds),np.array(grades))

trainproblems, traingrades = get_problem_set(train_filename)
testproblems, testgrades = get_problem_set(test_filename)

print('trainproblems', len(trainproblems))
print('traingrades',   len(traingrades))
print('testproblems',  len(testproblems))
print('testgrades',    len(testgrades))

np.savez(
    'moonboard.npz', x_train=trainproblems, x_test=testproblems,
    y_train=traingrades, y_test=testgrades)