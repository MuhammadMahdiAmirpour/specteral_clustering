import numpy as np

Xs = np.array([[1, 2, 3, 4, 5],
               [3, 4, 5, 6, 7],
               [5, 6, 7, 8, 9]])

Ys = np.array([[3, 4, 5, 6, 7],
               [8, 9, 10, 11, 12],
               [1, 2, 3, 4, 5]])

common_elements_matrix = np.array([[np.intersect1d(X, Y).size for Y in Ys] for X in Xs])
print(common_elements_matrix)

