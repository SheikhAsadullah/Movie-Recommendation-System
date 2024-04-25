#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity


# In[3]:


movies_df = pd.read_csv("movies.csv")
ratings_df = pd.read_csv("ratings.csv")


# In[4]:


df = pd.merge(movies_df, ratings_df, on='movieId')


# In[5]:


print("Objective 1:-\nCreate a popularity-based recommender system at a genre level. The user will input a genre (g), minimum rating threshold (t) for a movie, and no. of recommendations(N) for which it should be recommended top N movies which are most popular within that genre (g) ordered by ratings in descending order where each movie has at least (t) reviews.")


# In[6]:


genres = df["genres"].str.split("|").explode().unique()


# In[7]:


genre= input(f"Please enter the Genre out of {genres}: ")
threshold=int(input("Please enter the minimum rating threshold: "))
N=int(input("Please enter the number of movie recommendation: "))


# In[8]:


def objective1():
    gen_movie = df[df['genres'].str.contains(genre, case=False)]
    stats = gen_movie.groupby(['title']).agg({'rating': ['mean', 'count']})
    stats.columns = ['Average Movie Rating', 'Num Reviews']
    t_movie = stats[stats['Num Reviews'] >= threshold]
    result = t_movie.sort_values(by='Average Movie Rating', ascending=False).head(N).reset_index()
    return result


# In[9]:


print(f"Top {N} movies within {genre} genre which has atleast {threshold} reviews are :-\n",objective1())


# In[10]:


print("Objective 2:-\nCreate a content-based recommender system that recommends top N numbers movies based on similar movie genres watched by a random user. ")


# In[15]:


movie_title = input("Enter the movie name (Please mention the yaer of release too): ")
N = int(input("Enter the number of reccomendation required : "))


# In[16]:


def objective2():
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    idx = movies_df.index[movies_df['title'] == movie_title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:N+1]
    movie_indices = [i[0] for i in sim_scores]
    recommendations = movies_df['title'].iloc[movie_indices].tolist()
    #print("Recommendations for", movie_title)
    for i, title in enumerate(recommendations, 1):
        print(f"{i}. {title}")


# In[17]:


print(f"Top {N} movies similar to {movie_title} :-\n")
objective2()


# In[18]:


print("Objective 3:-\nCreate a collaborative based recommender system which recommends top N movies based on “K” similar users for a target user “u” ")


# In[19]:


user_id = int(input("Enter the User ID of the target user : "))
n = int(input("Enter the number of recommendation required : "))
k = int(input("Enter the similar user threshold : "))


# In[20]:


def objective3():
    matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    sim = cosine_similarity(matrix)
    sim_users = sim[user_id - 1].argsort()[::-1][1:k+1] + 1
    sim_users_movies = ratings_df[ratings_df['userId'].isin(sim_users)]
    filtered_movies = sim_users_movies[sim_users_movies['userId'] != user_id]
    group_ratings = sim_users_movies.groupby('movieId')['rating'].agg(['mean', 'count'])
    top_movies = group_ratings.sort_values(by=['mean', 'count'], ascending=False).head(n)
    recommended_movies=top_movies.index.tolist()
    print(f'Top {n} movies based on “{k}” similar users for a target user, "User ID :{user_id}"')
    for i, movie_id in enumerate(recommended_movies, 1):
        movie_title = movies_df.loc[movies_df['movieId'] == movie_id, 'title'].iloc[0]
        print(f"{i}. {movie_title}")


# In[21]:


objective3()


# In[ ]:




