import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Prints the number of rows, number of unique values for each attribute, and number of rows with missing values for each column

print("Loading dataframe")

#data_x = pd.read_pickle('data_x.pkl')
#data_y = pd.read_pickle('data_y.pkl')
df = pd.read_pickle('data_all.pkl')

print(list(df.columns))
print(df.shape[0], ' rows')
n = len(pd.unique(df['Customer_ID']))
print(n, " customers")
n = len(pd.unique(df['Movie_ID']))
print(n, " movies")
n = len(pd.unique(df[['Actor1', 'Actor2','Actor3']].values.ravel('K')))
print(n," actors")
n = len(pd.unique(df[['Director1', 'Director2']].values.ravel('K')))
print(n, " directors")
n = len(pd.unique(df[['Writer1', 'Writer2']].values.ravel('K')))
print(n," writers")
n = len(pd.unique(df[['Genre1', 'Genre2','Genre3']].values.ravel('K')))
print(n," genres")
n = len(pd.unique(df['Production_Company']))
print(n, " production companies")
n = len(pd.unique(df['Language']))
print(n, " languages")
n = len(pd.unique(df['Country']))
print(n, " countries")

print("Number of rows with null values:")
print(df.isnull().sum())