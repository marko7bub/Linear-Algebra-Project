import numpy as np
import pandas as pd
import project
import copy
import math
from random import random, randint
from matplotlib import pyplot as plt


# Read csv-file with ratings, that we got with 'file_preparation.py'
df = pd.read_csv('new_ratings2.csv')
df1 = pd.read_csv('movies_metadata.csv', low_memory=False)
# Set column of films as an index column
df = df.set_index('0')

arr = df.to_numpy()
# Get coordinates of nonzero values from 2D-array with ratings
coords = np.transpose(np.nonzero(arr)).tolist()

# rat_dict:
#    key: index of row with nonzero value; row represents some film
#    value: index of column; column represents some user
rat_dict = dict()
for i in coords:
    if i[0] not in rat_dict.keys():
        rat_dict[i[0]] = [i[1]]
    else:
        rat_dict[i[0]].append(i[1])

# Sort our films by number of reviews in descending order
sorted_keys = sorted(rat_dict, key=lambda x:len(rat_dict[x]), reverse=True)

def full_matr(n: int):
    # Returns full matrix of ratings
    films = [sorted_keys[0]]
    ind = 1
    # Intersection of users
    inter = []

    while len(inter) < n:
        try:
            inter = set(rat_dict[sorted_keys[0]]).intersection(rat_dict[sorted_keys[ind]])
            ind += 1
        except IndexError:
            print('IMPOSSIBLE')
            return

    while len(films) < n:
        try:
            if len(inter.intersection(set(rat_dict[sorted_keys[ind]]))) >= n:
                inter = inter.intersection(set(rat_dict[sorted_keys[ind]]))
                films.append(sorted_keys[ind])
            ind += 1
        except IndexError:
            print('IMPOSSIBLE')
            return

    # Create zero array
    rat_arr = np.zeros((n, n))
    films, inter = films[:n], list(inter)[:n]


    # Fill zero array with values from initial array
    for k in range(n):
        for j in range(n):
            rat_arr[k, j] = arr[films[k]][inter[j]]

    # print('fulms', films)
    key = 0

    lst = []
    for i in df.iterrows():
        if key in films:
            # print(i[0])
            lst.append(str(i[0]))
        key +=1
    print(lst)

    titles = [0 for i in range(n)]
    df2 = pd.read_csv('movies_metadata.csv', low_memory=False)
    df3 = pd.read_csv('links.csv', low_memory=False, dtype=str)

    lst_imdb = [0 for i in range(n)]
    for row in df3.iterrows():
        if row[1][0] in lst:
            lst_imdb[lst.index(row[1][0])] = row[1][1]

    print(lst_imdb)

    for row in df2.iterrows():
        if str(row[1][6])[2:] in lst_imdb:
            titles[lst_imdb.index(str(row[1][6])[2:])] = row[1][8]
            
    print(titles)

    return rat_arr, titles

full_matrix = full_matr(18)[0]

def make_null(data):
    new_data = copy.deepcopy(data)
    for i in new_data:
        random_index_1 = randint(0, 17)
        random_index_2 = randint(0, 17)
        random_index_3 = randint(0, 17)
        random_index_4 = randint(0, 17)
        random_index_5 = randint(0, 17)
        random_index_6 = randint(0, 17)
        random_index_7 = randint(0, 17)
        random_index_8 = randint(0, 17)
        random_index_9 = randint(0, 17)


        

        i[random_index_1] = 0
        i[random_index_2] = 0
        i[random_index_3] = 0
        i[random_index_4] = 0
        i[random_index_5] = 0
        i[random_index_6] = 0
        i[random_index_7] = 0
        i[random_index_8] = 0
        i[random_index_9] = 0


    return new_data

sparse_matrix = make_null(full_matrix)

print(full_matrix)
print()
print(sparse_matrix)
print()
predicted_matrix, p, q = project.fill_the_ratings_normal(sparse_matrix, 2, 0.01, 0.02, 10000)
print(predicted_matrix)
print()
rmse = math.sqrt((np.square(full_matrix - predicted_matrix)).mean())
print("RMSE: ", rmse)
print()
