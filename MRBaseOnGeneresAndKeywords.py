import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import data from csv
dfKeywords = pd.read_csv('keywords.csv')
dfMetadata = pd.read_csv('movies_metadata.csv', low_memory=False)
dfRating = pd.read_csv('ratings.csv')
#=============#PREPARE DATA#=============#
#Changing column id name to movieId
dfKeywords.rename(columns={'id':'movieId'},inplace=True)

# Creating new dataframe without specific columns
dfMetadata.rename(columns={'id':'movieId'}, inplace=True)
dfMetadata = dfMetadata.drop(['adult','belongs_to_collection'
                ,'budget','homepage'
                ,'original_language','original_title'
                ,'poster_path','production_companies'
                ,'production_countries','release_date'
                ,'revenue','runtime'
                ,'spoken_languages','status'
                ,'tagline','video'], axis=1)
dfMetadata = dfMetadata.dropna(subset=['title'])
dfMetadata['popularity'] = pd.to_numeric(dfMetadata['popularity'], errors='coerce', downcast='integer')

# Change movieId to integer in both dataFrame
dfMetadata['movieId'] = pd.to_numeric(dfMetadata['movieId'], errors='coerce', downcast='integer')
dfRating['movieId'] = pd.to_numeric(dfRating['movieId'], errors='coerce', downcast='integer')

# Prepare ratings with possible Ids
MetadatalistId = pd.to_numeric(dfMetadata['movieId'], errors='coerce', downcast='integer')
dfRating = dfRating[dfRating.movieId.isin(MetadatalistId)]
dfRating = dfRating.drop(['userId','timestamp'], axis=1)

# Merge all in one DataFrame
dfRating = dfRating.merge(dfMetadata,on='movieId', how='inner')
dfRating = dfRating.merge(dfKeywords,on='movieId', how='inner')

# Corecting 'vote_average' and 'vote_count' to real values
dfRating['vote_count'] = dfRating.groupby(['movieId'])['movieId'].transform('count')
dfRating['vote_average'] = dfRating.groupby(['movieId'])['rating'].transform('mean')
dfRating = dfRating.drop_duplicates(subset='movieId')
dfRating = dfRating.drop(['rating'],axis=1)
# dfRating.to_csv('dfRating.csv')

# Ploting first ten popular movies
popularity = dfRating.sort_values('popularity', ascending=False)
plt.figure(figsize=(15,5))
plt.barh(popularity['title'].head(10), popularity['popularity'].head(10), align='center', color=(0.4,0.5,0.6,0.8))
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.ylabel("Title")
plt.title("Popular movies")
# plt.show()

#=============#RECOMMENDATION SYSTEM#=============#
# System recommendation base on keywords and genre 

# Function converting all string to lower case and without spaces
def convert_string(s):
    if isinstance(s,list):
        return [str.lower(i.replace(" ","")) for i in s]
    else:
        if isinstance(s, str):
            return str.lower(s.replace(" ",""))
        else:
            return ''

# Function returning list of top 3 elements or return whole list 
def get_list(ls):
    if isinstance(ls, list):
        elements = [i['name'] for i in ls]

        if len(elements) > 3:
            elements = elements[:3]
        return elements
    
    return []

# Function combine keywords and genres
def create_combination(row):
    return ' '.join(row['genres'])+' '.join(row['keywords'])

# Function creating recommendtation base on title user write
def get_recommendations(title, cosine_sim):
    try:
        idx = indices[title]
    except KeyError:
        print('Sorry we don\'t have this title in our database. Try another one')
        print('Here\'s few available titles: ')
        print(indices.sample(10))
        return None
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)
    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]

    return dfRating['title'].iloc[movie_indices]

columns = ['keywords', 'genres']

# Create list of keywords and generes easy to use 
for column in columns:
    dfRating[column] = dfRating[column].apply(literal_eval)
    dfRating[column] = dfRating[column].apply(get_list)
    dfRating[column] = dfRating[column].apply(convert_string)

dfRating['combination'] = dfRating.apply(create_combination, axis=1)

# Saving result to csv 
# dfRating.to_csv('movie_rating.csv')

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(dfRating['combination'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

dfRating = dfRating.reset_index()
indices = pd.Series(dfRating.index, index=dfRating['title'])

#=============#TESTING SOLUTION#=============#
title = input('Type title to get recommendation: ')
print('Your recommendation',get_recommendations(title, cosine_sim2))
