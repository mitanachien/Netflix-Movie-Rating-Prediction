import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_pickle('data_all.pkl')

print("Normalizing data")

# Normalize year and duration using min-max scaling
df = df.astype({'Year': 'float', 'Duration':'float'})
column = 'Year'
df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())  
column = 'Duration'
df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())    


unique_ID = df['Customer_ID'].unique()
count = 0
training_df = pd.DataFrame()
testing_df = pd.DataFrame()

print("Splitting data into training and test")

for id in unique_ID:
    if (count < 5000):
        filtered_df = df[df['Customer_ID']==id]
        train_temp = filtered_df.sample(frac = 0.5)
        test_temp = filtered_df.drop(train_temp.index)
        training_df = training_df.append(train_temp, ignore_index=True)
        testing_df = testing_df.append(test_temp, ignore_index=True)
        count+=1
        if count%100 == 0:
            print('Process {0:d} times'.format(count))


training_df.to_pickle('training.pkl')
testing_df.to_pickle('test.pkl')

print("Training data set: ", training_df.shape)
print("Test data set: ", testing_df.shape)
print("Complete")

