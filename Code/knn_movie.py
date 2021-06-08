# Note: run avg_movie_rating.py first

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from scipy import stats
from statistics import mean
from sklearn.metrics import accuracy_score

print("Loading dataframe")
training = pd.read_pickle('training.pkl')
test = pd.read_pickle('test.pkl')
avg_ratings = pd.read_pickle('avg_ratings.pkl')

#######################

def distance_movies(x1, x2):
    dist = 0
    x2_actors = {x2['Actor1'], x2['Actor2'], x2['Actor3']}
    if not (x1['Actor1'] in x2_actors):
        dist += 1
    if not (x1['Actor2'] in x2_actors):
        dist += 1
    if not (x1['Actor3'] in x2_actors):
        dist += 1
    if not (x1['Country'] == x2['Country']):
        dist += 1
    x2_directors = {x2['Director1'], x2['Director2']}
    if not(x1['Director1'] in x2_directors):
        dist += 1
    if not(x1['Director2'] in x2_directors):
        dist += 1
    x2_genres = {x2['Genre1'], x2['Genre2'], x2['Genre3']}
    if not(x1['Genre1'] in x2_genres):
        dist += 1
    if not(x1['Genre2'] in x2_genres):
        dist += 1
    if not(x1['Genre3'] in x2_genres):
        dist += 1
    if not (x1['Language'] == x2['Language']):
        dist += 1
    if not (x1['Production_Company'] == x2['Production_Company']):
        dist += 1
    x2_writers = {x2['Writer1'], x2['Writer2']}
    if not(x1['Writer1'] in x2_writers):
        dist += 1
    if not(x1['Writer2'] in x2_writers):
        dist += 1
    dist =+ np.square(x1['Year'] - x2['Year'])
    dist =+ np.square(x1['Duration'] - x2['Duration'])
    return np.sqrt(dist)

def calculate_distances(row, dataset):
    list = []
    length = dataset.shape[0]
    for i in range(0, length):
        row2 = dataset.iloc[i,:]
        list.append(distance_movies(row,row2))
    return np.asarray(list)

##########################

def knn_movie_similarity(train, test, k):
    predicted_labels = []
    test_length = test.shape[0]
    for i in range(0, test_length):
        test_row = test.iloc[i,:]
        customer_id = test_row['Customer_ID']
        # look only at movies that customer has rated
        customer_train = train[train['Customer_ID']==customer_id]
        if(customer_train.shape[0] > 0):
            # find closest k movies, and average their ratings
            distances = calculate_distances(test_row, customer_train)
            sorted_indices = np.argsort(distances)
            nearest_indices = sorted_indices[0:k]
            neighbor_ratings = []
            for index, index_of_nn in enumerate(nearest_indices):
                neighbor_ratings.append(customer_train.iloc[index_of_nn].loc['Rating'])
            predicted_labels.append(mean(neighbor_ratings))
        else: # if the customer has not rated any movies yet, just predict the average rating for the movie across all users
            avg_row = avg_ratings[avg_ratings['Movie_ID']==test_row['Movie_ID']]
            if avg_row.empty:
                predicted_labels[index] = 3
            else:
                avg = avg_row['Avg_Rating']
                predicted_labels.append(avg)
    return np.asarray(predicted_labels)

##########################

test = test.sample(200, random_state = 42)
print(len(training['Customer_ID'].unique()), " customers in training")
print(len(test['Customer_ID'].unique()), " customers in test")
print("Training shape: ", training.shape)
print("Test shape: ", test.shape)

print("Performing KNN Regression with movie distance metric")

start = time.time()
predictions = knn_movie_similarity(training, test, 5)
rounded_predictions = np.around(predictions)
end = time.time()
print(end - start, " seconds elapsed")
actual = test['Rating'].to_numpy()
rmse = np.sqrt(np.square(rounded_predictions - actual).mean())
print("RMSE: ", rmse)

cm = confusion_matrix(actual, rounded_predictions)
print(cm)
accuracy = accuracy_score(actual, rounded_predictions)
print("Accuracy: ", accuracy)

print("Saving predictions")
np.save('knn_movie_predictions.npy', predictions)

print("Done!")