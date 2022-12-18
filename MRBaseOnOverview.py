import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# Import data from csv
df1 = pd.read_csv('IMDB/tmdb_5000_credits.csv')
df2 = pd.read_csv('IMDB/tmdb_5000_movies.csv')
#=============#PREPARE DATA#=============#
df1.columns=['id', 'title', 'cast', 'crew']
df2.merge(df1, on='id')

C = df2['vote_average'].mean()
m = df2['vote_count'].quantile(0.9)

q_movies = df2.copy().loc[df2['vote_count'] >= m]
# q_movies = q_movies.dropna(subset=['homepage'])
def weight_rating(tab, m=m, C=C):
    '''
    Weighted Rating by IMDB
    Parameters
    ----------
    v : is the number of votes for the movie
    m : is the minimum votes required to be listed in the chart
    R : is the average rating of the movie 
    C : is the mean vote across the whole report
    Returns
    -------
    Calculation based on the IMDB formula
    '''
    v = tab['vote_count']
    R = tab['vote_average']
    return (v/(v+m)*R + m/(v+m)*C)

q_movies['score'] = q_movies.apply(weight_rating, axis=1)
q_movies = q_movies.sort_values('score', ascending=False)

tfidf = TfidfVectorizer(stop_words='english')
df2['overview'] = df2['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(df2['overview'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
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

    return df2['title'].iloc[movie_indices]

# Ploting first ten popular movies
popularity = q_movies.sort_values('popularity', ascending=False)
plt.figure(figsize=(15,5))
plt.barh(popularity['title'].head(10), popularity['popularity'].head(10), align='center', color=(0.4,0.5,0.6,0.8))
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.ylabel("Title")
plt.title("Popular movies")
# plt.show()

#=============#TESTING SOLUTION#=============#
title = input('Type title to get recommendation: ')
print('Your recommendation\n', get_recommendations(title))
