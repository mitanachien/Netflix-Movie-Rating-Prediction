import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats
from statistics import mean

print("Loading dataframe")
df = pd.read_pickle('data_all.pkl')

print("Normalizing data")
# Normalize year and duration using min-max scaling
df_normalized = df.copy()
df_normalized = df_normalized.astype({'Year': 'float', 'Duration':'float'})
column = 'Year'
df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
column = 'Duration'
df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())    

print("Splitting into training, validation, and test sets")
# Split data set into training, validation, and test (0.6, 0.2, 0.2)
train, val_test = train_test_split(df_normalized, test_size=0.4, random_state=1)
val, test = train_test_split(val_test, test_size=0.5, random_state=1)

def distance(x1, x2):
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
        list.append(distance(row,row2))
    return np.asarray(list)

def knn(train, test, k):
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
                neighbor_ratings.append(train.iloc[index_of_nn].loc['Rating'])
            predicted_labels.append(mean(neighbor_ratings))
        else:
            predicted_labels.append(3)
    return np.asarray(predicted_labels)

print("Performing KNN Regression")
small_train = train[train['Customer_ID'].between(1,500)]
small_val = val[val['Customer_ID'].between(1,500)]
predictions = knn(small_train, small_val, 5)
actual = small_val['Rating'].to_numpy()
rmse = np.sqrt(np.square(predictions - actual).mean())
print("RMSE: ", rmse)