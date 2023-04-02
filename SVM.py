import numpy as np
from load_moonboard import load_moonboard
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

grades = np.array([4, 10])

# Load data into numpy array

(x_train, y_train), (x_test, y_test) = load_moonboard(grades=[4, 7, 10], n_data=1000)
x_train = x_train.reshape(-1, 18 * 11)
x_test = x_test.reshape(-1, 18 * 11)

pca = PCA(n_components=20)
x_train = pca.fit_transform(x_train, y_train)
x_test = pca.transform(x_test)

svm = SGDClassifier(loss='hinge', penalty=..., alpha=..., max_iter=1000, learning_rate=..., eta0=...,
                    random_state=3);

param_grid = {'penalty': ['l1', 'l2'],
              'alpha': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
              'learning_rate': ['constant', 'invscaling'],
              'eta0': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]  # write you code here
              };

grid = GridSearchCV(estimator=svm, param_grid=param_grid, refit=True, verbose=0);  # you will need to play with that

grid.fit(x_train, y_train);

grid_predictions = grid.predict(x_test);

print(classification_report(y_test,
                            grid_predictions))
print(grid.best_params_, grid.best_score_)

plt.figure(figsize=(10,5))
plt.plot(x_train[y_train == 0,0],x_train[y_train == 0,1],'.',label ='Class 1');
plt.plot(x_train[y_train == 1,0],x_train[y_train == 1,1],'ro',label ='Class 2');
#plt.plot(x_train[y_train==2,0],x_train[y_train==2,1],'k+',label ='Class 3');
plt.legend()
plt.show()