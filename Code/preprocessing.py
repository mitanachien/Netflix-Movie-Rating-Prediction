
import pandas as pd
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

### Add the movie_id's into each row for combined_data_1, 2, 3, and 4
for i in range(1,5):
    f1 = open(dir_path+'/data/combined_data_'+ str(i) +'.txt','r')
    f2 = open(dir_path+'/data/combined_data_cleaned_'+ str(i) +'.txt','w')
    cur_movie_id = f1.readline().strip()[:-1]
    for line in f1:
        parts = line.split(',')
        if len(parts) == 1:
            cur_movie_id = line.strip()[:-1]
            continue
        else:
            f2.write(','.join([cur_movie_id] + parts))
    f1.close()
    f2.close()

# convert files from txt to csv
for i in range(1,5):
    read_file = pd.read_csv(dir_path+'/data/combined_data_cleaned_'+ str(i) +'.txt', header = None)
    read_file.columns = ['movie_id','customer_id','rating', 'date']
    read_file.to_csv (dir_path+'/data/data_'+ str(i) +'.csv', index=None)

print("Done!")