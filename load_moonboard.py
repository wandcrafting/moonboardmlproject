import numpy as np

def load_moonboard(filename='moonboard.npz', grades = [4, 5, 6, 7, 8, 9, 10], n_data=None):
    f = np.load(filename)
    x_train, y_train = f['x_train'], f['y_train']
    x_test,  y_test  = f['x_test'],  f['y_test']

    grades = np.array(grades)

    grade_in_train = np.isin(y_train, grades)
    x_train = x_train[grade_in_train]
    y_train = y_train[grade_in_train]
    for i in range(len(np.unique(y_train))):
        y_i = np.unique(y_train)[i]
        y_train[y_train == y_i] = i

    grade_in_test = np.isin(y_test, grades)
    x_test = x_test[grade_in_test]
    y_test = y_test[grade_in_test]
    for i in range(len(np.unique(y_test))):
        y_i = np.unique(y_test)[i]
        y_test[y_test == y_i] = i

    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_moonboard() 

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

