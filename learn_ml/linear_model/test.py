from learn_ml.linear_model.perceptron import Perceptron
import numpy as np

X = np.array([[1, 3, 1], [2, 5, 1], [3, 8, 1], [2, 6, 1], [3, 1, -1], [4, 1, -1], [6, 2, -1], [7, 3, -1]])
y = np.array([1, 1, 1, 1, -1, -1, -1, -1])

clf = Perceptron(eta=0.5, epoch=10000)
clf.fit(X, y)
y_pred = clf.predict(np.array([[1, 3, 1]]))
print(y_pred)

print(clf.loss(X,y))