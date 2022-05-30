import numpy as np
import math

data1 = np.array([
    [2, 5, 0, 2],
    [0, 4, 0, 5],
    [3, 0, 1, 5],
    [1, 0, 5, 0],
    [0, 2, 3, 2],
])

def fill_the_ratings(data, k, alpha, lambd, epochs):

    # Initialize the P and Q matrices with k latent features
    # and fill it with values of normal distribution

    q = np.random.normal(0, 0.1, (len(data), k))
    p = np.random.normal(0, 0.1, (len(data[0]), k))    

    availible_ratings = [(i, j, data[i, j])
                        for i in range(len(data))
                        for j in range(len(data[0]))
                        if data[i, j] > 0]
    
    for epoch in range(epochs):
        for i, j, rating in availible_ratings:
            difference = rating - np.dot(q[i], p[j])
            q[i] += alpha * (difference * p[j] - lambd*q[i])
            p[j] += alpha * (difference * q[i] - lambd*p[j])

    return q.dot(p.T), q, p

def fill_the_ratings_same(data, k, alpha, lambd, epochs):

    #Initialize the P and Q matrices with k latent features and fill it with same values

    p = np.full((len(data), k), 0.1)
    q = np.full((len(data[0]), k), 0.1)    

    availible_ratings = [(i, j, data[i, j])
                        for i in range(len(data))
                        for j in range(len(data[0]))
                        if data[i, j] > 0]
    
    for epoch in range(epochs):
        for i, j, rating in availible_ratings:
            difference = rating - np.dot(q[i], p[j])
            q[i] += alpha * (difference * p[j] - lambd*q[i])
            p[j] += alpha * (difference * q[i] - lambd*p[j])

    return q.dot(p.T), q, p
    
print()
print("Initial matrix:")
print()
print(data1)
print()
print("Predicted matrix with approximated values")
print()
print(fill_the_ratings(data1, 2, 0.01, 0.02, 10000)[0])
print()

