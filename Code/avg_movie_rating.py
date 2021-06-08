# Calculate the average rating for each movie (in the training set) and save it in a dataframe
# Note: run data_splitting first

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats
from statistics import mean

print("Loading dataframe")

df = pd.read_pickle('training.pkl')
train = df[['Movie_ID', 'Rating']]

print("Calculate averages")

avg_ratings = pd.DataFrame(columns = ['Movie_ID', 'Avg_Rating'])

# Calculate average rating for each movie in the training set
unique_ID = df['Movie_ID'].unique()
for i in unique_ID:
    rows = train[train['Movie_ID']==i]
    ratings = rows['Rating'].values
    avg = ratings.mean()
    avg_ratings = avg_ratings.append({'Movie_ID' : i, 'Avg_Rating' : avg}, ignore_index = True)

print("Saving dataframe")

# Pickle dataframe containing averages
avg_ratings.to_pickle('avg_ratings.pkl')

print("Done!")