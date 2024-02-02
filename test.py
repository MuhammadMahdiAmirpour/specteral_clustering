import numpy as np

Xs = np.array([[1, 2, 3, 4, 5],
               [3, 4, 5, 6, 7],
               [5, 6, 7, 8, 9]])

Ys = np.array([[3, 4, 5, 6, 7],
               [8, 9, 10, 11, 12],
               [1, 2, 3, 4, 5]])

# Reshape Xs and Ys to have 3 dimensions
Xs_reshaped = Xs[:, None, :]
Ys_reshaped = Ys[None, :, :]

# Use broadcasting to create the common_elements_matrix
common_elements_matrix = np.sum(np.isin(Xs_reshaped, Ys_reshaped), axis=-1)
print(common_elements_matrix)


common_elements_matrix = np.array([[np.intersect1d(X, Y).size for Y in Ys] for X in Xs])

print(common_elements_matrix)

