#!/usr/bin/env python
# coding: utf-8

# # Video Game Recommendation Engine
# 
# 

# ### Overview:
# 
# Welcome to my project on creating a video game recommendation system. Many streaming services utilize recommendation systems to increase customer engagement with their platform. I wanted to create a similar system for video games to display new games for users to play. In this project, we will be using a content-based recommender system. Therefore, we will base our recommendations on titles, publishers, descriptions, genres, and tags that different items share. During this project, I will be utilizing the packages Pandas, Numpy, and Sklearn. These are all standard packages for data manipulation, mathematics, and machine learning applications.
# 
# Link for Dataset: https://www.kaggle.com/trolukovich/steam-games-complete-dataset

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


Games = pd.read_csv('~/Downloads/steam_games 2.csv')


# ### Background
# The dataset features 20 columns, many that will not be of use to this type of recommendation system. As well, there are 40,833 unique video games with unique characteristics. The recommendation system is designed to suit the needs of novice gamers. Therefore, we will be excluding free games and focusing on Triple-A titles. Triple-A games are video games produced or developed by a major publisher, which allocated a large budget for both development and marketing. Many novice gamers will be familiar with Triple-A games rather than small indie games. Most Triple-A titles retail price is \\$59.99, however, some games release months or years after their console release to the steam platform for a discount. Therefore we will limit our dataset to only titles with a price range of \\$19.99 to \\$59.99.

# In[3]:


Games.head(3)


# ### Step One: Filtering the price between \\$19.99 and \\$59.99
# The original price column will be the column we intend to filter. We have a problem to sort out before we proceed with our filtering. We cannot sort the original price column because it is not considered a numerical type. We can fix this by first converting the column to a character type, then remove the dollar sign through character string slicing. After we remove the dollar sign, we can convert the column to a numerical type. Now we can proceed with applying the filter. The total number of unique games in the dataset is now 4,338.

# In[4]:


Games.original_price


# In[5]:


Games['original_price'] = Games['original_price'].str[1:]


# In[6]:


Games['original_price'] = pd.to_numeric(Games['original_price'],errors='coerce')


# In[7]:


Games = Games[(Games['original_price'] >= 19.99) & (Games['original_price'] <= 59.99)]


# In[8]:


Games.shape


# ### Step Two: Choosing columns to use in the recommendation system
# When choosing which columns to put in the recommendation system, we should be mindful of the characteristics gamer's value. The developer variable is important to include since developers often have the same team working on different games. Therefore each game produced by the same developer will have a similar style of gameplay. Genre variable provides a broad grouping of games with similarities in form, style, or subject matter. Popular Tags variable is an in-depth description of different gaming characteristics. The Game Details variable lists a game's online offering such as whether a game is single-player or multiplayer. The last variable would be the name of the game, which is valuable because sequels and prequels will be included in the recommendation.

# In[9]:


Games.head(3)


# In[10]:


Games = Games[['genre','game_details','popular_tags','developer','name']]


# ### Step Three: Drop all rows with null values
# Usually, the first step in any project would be to eliminate null values. However, it is important to wait to perform this step. We have previously consolidated columns to only useful columns for the recommendation system. Now that the dataset only has useful columns, we can eliminate only rows where null values are present in the columns we have chosen. After eliminating null values the total unique games in the dataset are 3,999. We will also be adding a new column labeled Game_ID, which provides a numerical unique value to each game. 

# In[11]:


Games.head(3)


# In[12]:


Games.dropna(inplace = True)


# In[13]:


Games.shape


# In[14]:


Games['Game_ID'] = range(0,3999)


# In[15]:


Games.isnull().values.any()


# In[16]:


Games = Games.reset_index()


# ### Step Four: Combine selected column's values into string
# Our next step is going to be creating a function that compiles all data in each column selected into one giant string. In order to do so, we are going to make an empty list called important features and then append the values of the desired columns. Then we create a column called important features, where we call the function on the dataset.

# In[17]:


def get_important_features(data):
    important_features = []
    for i in range(0, data.shape[0]):
        important_features.append(data['name'][i]+' '+data['developer'][i]+' '+data['popular_tags'][i]+' '+data['genre'][i]+data['game_details'][i])
        
    return important_features


# In[18]:


Games['important_features'] = get_important_features(Games)
Games.important_features.head(3)


# ### Step Five: Assemble similarity matrix
# First, we will be using the count vectorizer function to transform a given text into a vector. The matrix consists of a frequency of words in a string. For example the string 'Action, Action, Adventure', the matrix will display a table with the word, Action, and a frequency of two. Then we can use the cosine similarity function to measure the correlation among the different games. This function produces a matrix with the correlations between each game. The matrix contains a numerical value from 0 to 1, where a variable closer to 1 is considered a good recommendation, and a variable closer to 0 is considered a poor recommendation. The diagonal line of the value 1 showcases a perfect correlation because it is the same game on each axis.

# In[19]:


cm = CountVectorizer().fit_transform(Games['important_features'])


# In[20]:


cs = cosine_similarity(cm)


# In[21]:


print(cs)


# ### Step Six: Use the Recommendation System
# Our last step would be to enter the name of the game we wish to get recommendations from. In this case, I have chosen the game Doom Eternal. We then create a new object called title_id, where we obtain the Game_ID value for Doom Eternal, which we assigned to each title in Step 3. After this step, we are going to create a list of enumerations that contain the similarity score between each game and Doom Eternal. Then we sort the similarity score in descending order to receive the games with the highest similarities to Doom Eternal. I have chosen to display the top 7 games that are recommended to us based on the characteristics of Doom Eternal.

# In[22]:


title = 'DOOM Eternal'
title_id = Games[Games.name == title]['Game_ID'].values[0]


# In[23]:


scores = list(enumerate(cs[title_id]))


# In[24]:


sorted_scores = sorted(scores, key = lambda x:x[1], reverse = True)
sorted_scores = sorted_scores[1:]


# In[25]:


j = 0
print('The 7 most recommended games to', title, 'are:\n')
for item in sorted_scores:
    game_title = Games[Games.Game_ID == item[0]]['name'].values[0]
    print(j+1, game_title)
    j = j+1
    if j > 6:
        break


# ### Conclusion
# 
# When observing the top seven results we can see the similarities between the games. The more similarities in each column the higher the ranking will be. For instance, Doom 3: BFG Edition and DOOM have similarities in every column. While the bottom four recommendations have values in common in the genre, game details, and popular tags columns. From my personal experience playing five out of the seven recommended games, I would like to have these games recommended to me based on my interest of DOOM Eternal.

# In[26]:


Games = Games.set_index('name')


# In[27]:


Games.loc[['DOOM Eternal','Doom 3: BFG Edition','DOOM','Dead Space™ 2','DUSK','Max Payne 3','Unreal Tournament 3 Black','Crysis 2 - Maximum Edition'],
         ['genre','game_details','popular_tags','developer']]

