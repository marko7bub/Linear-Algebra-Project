import pandas as pd

df = pd.read_csv('ratings_small.csv')
# Create new dataframe with movieId's an indexes and userId's as columns
rating = pd.DataFrame(0, index=list(set(df['movieId'].tolist())), columns=list(set(df['userId'].tolist())))

# Fill empty dataframe with ratings
for index, row in df.iterrows():
    rating.loc[row['movieId'], row['userId']] = row['rating']

rating.to_csv('new_ratings2.csv')
