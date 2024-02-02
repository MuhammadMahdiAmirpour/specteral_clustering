import numpy as np
import math
from sklearn.metrics import adjusted_rand_score

def find_equal_indices(array):
    """
  Finds the indices of elements that appear multiple times in a NumPy array.

  Args:
    array: A NumPy array.

  Returns:
    A list of lists, where each sublist contains the indices of an element that appears
    multiple times in the array.
  """

    # Use numpy.where to find the indices of each element.
    # unique, counts = np.unique(array, return_counts=True)
    # multiple_indices = [np.where(array == value)[0].tolist() for value in unique[counts > 0]]
    # Get unique values and their counts
    unique, counts = np.unique(array, return_counts=True)

    # Find indices for values with counts greater than 1 using broadcasting
    mask = counts > 1
    multiple_indices = np.split(np.where(array[:, None] == unique[mask])[1], np.cumsum(counts[mask])[:-1])


    return multiple_indices


def table_c(X, Y):
    xs = find_equal_indices(X)
    ys = find_equal_indices(Y)
    T = np.zeros((len(xs), len(ys)))
    for i in range(len(xs)):
        for j in range(len(ys)):
            T[i, j] = len(set(xs[i]).intersection(ys[j]))
    A = np.sum(T, axis=0)
    B = np.sum(T, axis=1)
    return T, A, B

def RI(T):
    sum = 0
    for i in range(len(T)):
        for j in range(len(T[0])):
            sum += math.comb(int(T[i][j]), 2)

    return sum

def exp(A, B, n):
    a = 0
    for x in A:
        a += math.comb(int(x), 2)
    b = 0
    for x in B:
        b += math.comb(int(x), 2)
    return a * b / math.comb(n, 2)

def max(A, B):
    a = 0
    for x in A:
        a += math.comb(int(x), 2)
    b = 0
    for x in B:
        b += math.comb(int(x), 2)
    return (a + b) / 2


def ARI(X, Y):
    n = len(X)
    T, A, B = table_c(X, Y)
    return (RI(T) - exp(A, B, n)) / (max(A, B) - exp(A, B, n))

if __name__ == "__main__":
    # Example usage:
    true_labels = np.array([0, 0, 1, 1, 1, 2, 2, 2])
    predicted_labels = np.array([0, 0, 1, 1, 2, 2, 2, 2])
    print(ARI(true_labels, predicted_labels))
    # Example usage
    # array = np.array([1, 2, 2, 2, 3, 3, 1, 0, ])
    # result = find_equal_indices(array)
    # print(result)  # Output: [[0, 6], [1, 2, 3], [4, 5]]


