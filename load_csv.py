import pandas as pd
import numpy as np

print("Converting csv to pandas dataframe...")
df = pd.read_csv ('Netflix_with_IMDB_with_customerID.csv', usecols=['Actor1', 'Actor2', 'Actor3', 'Country', 'Customer Id', 'Director1', 'Director2', 'Genre1', 
                                                                    'Genre2', 'Genre3', 'Language', 'Movie ID', 'Production Company', 'Title', 'Writer1', 
                                                                    'Writer2', 'Year Of Release', 'Duration', 'Rating'],
                  encoding='utf-8', dtype={'Actor1':str, 'Actor2':str, 'Actor3':str, 'Country':str, 'Customer Id':np.uint32, 'Director1':str, 'Director2':str, 
                                           'Genre1':str, 'Genre2':str, 'Genre3':str, 'Language':str, 'Movie ID':np.uint16, 'Production Company':str, 'Title':str, 
                                           'Writer1':str, 'Writer2':str, 'Year Of Release':np.uint16, 'Duration':np.uint8, 'Rating':np.uint8})

# Rename columns to use underscores instead of spaces, for easier reference in functions
df.rename(columns={"Movie ID": "Movie_ID", "Customer Id": "Customer_ID", "Production Company":"Production_Company", "Year Of Release":"Year"}, inplace = True)

data_y = df['Rating'].values
data_x = df.drop(['Rating'], axis=1).values

# Save dataframe for later use
df.to_pickle('data_x.pkl')
df.to_pickle('data_y.pkl')
df.to_pickle('data_all.pkl')


print(list(df.columns))
print("Done!")
