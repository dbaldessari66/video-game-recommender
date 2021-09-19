# Video Game Recommendation Engine



### Overview:

Welcome to my project on creating a video game recommendation system. Many streaming services utilize recommendation systems to increase customer engagement with their platform. I wanted to create a similar system for video games to display new games for users to play. In this project, we will be using a content-based recommender system. Therefore, we will base our recommendations on titles, publishers, descriptions, genres, and tags that different items share. During this project, I will be utilizing the packages Pandas, Numpy, and Sklearn. These are all standard packages for data manipulation, mathematics, and machine learning applications.

Link for Dataset: https://www.kaggle.com/trolukovich/steam-games-complete-dataset


```python
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
```


```python
Games = pd.read_csv('~/Downloads/steam_games 2.csv')
```

### Background
The dataset features 20 columns, many that will not be of use to this type of recommendation system. As well, there are 40,833 unique video games with unique characteristics. The recommendation system is designed to suit the needs of novice gamers. Therefore, we will be excluding free games and focusing on Triple-A titles. Triple-A games are video games produced or developed by a major publisher, which allocated a large budget for both development and marketing. Many novice gamers will be familiar with Triple-A games rather than small indie games. Most Triple-A titles retail price is \\$59.99, however, some games release months or years after their console release to the steam platform for a discount. Therefore we will limit our dataset to only titles with a price range of \\$19.99 to \\$59.99.


```python
Games.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>types</th>
      <th>name</th>
      <th>desc_snippet</th>
      <th>recent_reviews</th>
      <th>all_reviews</th>
      <th>release_date</th>
      <th>developer</th>
      <th>publisher</th>
      <th>popular_tags</th>
      <th>game_details</th>
      <th>languages</th>
      <th>achievements</th>
      <th>genre</th>
      <th>game_description</th>
      <th>mature_content</th>
      <th>minimum_requirements</th>
      <th>recommended_requirements</th>
      <th>original_price</th>
      <th>discount_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://store.steampowered.com/app/379720/DOOM/</td>
      <td>app</td>
      <td>DOOM</td>
      <td>Now includes all three premium DLC packs (Unto...</td>
      <td>Very Positive,(554),- 89% of the 554 user revi...</td>
      <td>Very Positive,(42,550),- 92% of the 42,550 use...</td>
      <td>May 12, 2016</td>
      <td>id Software</td>
      <td>Bethesda Softworks,Bethesda Softworks</td>
      <td>FPS,Gore,Action,Demons,Shooter,First-Person,Gr...</td>
      <td>Single-player,Multi-player,Co-op,Steam Achieve...</td>
      <td>English,French,Italian,German,Spanish - Spain,...</td>
      <td>54.0</td>
      <td>Action</td>
      <td>About This Game Developed by id software, the...</td>
      <td>NaN</td>
      <td>Minimum:,OS:,Windows 7/8.1/10 (64-bit versions...</td>
      <td>Recommended:,OS:,Windows 7/8.1/10 (64-bit vers...</td>
      <td>$19.99</td>
      <td>$14.99</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://store.steampowered.com/app/578080/PLAY...</td>
      <td>app</td>
      <td>PLAYERUNKNOWN'S BATTLEGROUNDS</td>
      <td>PLAYERUNKNOWN'S BATTLEGROUNDS is a battle roya...</td>
      <td>Mixed,(6,214),- 49% of the 6,214 user reviews ...</td>
      <td>Mixed,(836,608),- 49% of the 836,608 user revi...</td>
      <td>Dec 21, 2017</td>
      <td>PUBG Corporation</td>
      <td>PUBG Corporation,PUBG Corporation</td>
      <td>Survival,Shooter,Multiplayer,Battle Royale,PvP...</td>
      <td>Multi-player,Online Multi-Player,Stats</td>
      <td>English,Korean,Simplified Chinese,French,Germa...</td>
      <td>37.0</td>
      <td>Action,Adventure,Massively Multiplayer</td>
      <td>About This Game  PLAYERUNKNOWN'S BATTLEGROUND...</td>
      <td>Mature Content Description  The developers de...</td>
      <td>Minimum:,Requires a 64-bit processor and opera...</td>
      <td>Recommended:,Requires a 64-bit processor and o...</td>
      <td>$29.99</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://store.steampowered.com/app/637090/BATT...</td>
      <td>app</td>
      <td>BATTLETECH</td>
      <td>Take command of your own mercenary outfit of '...</td>
      <td>Mixed,(166),- 54% of the 166 user reviews in t...</td>
      <td>Mostly Positive,(7,030),- 71% of the 7,030 use...</td>
      <td>Apr 24, 2018</td>
      <td>Harebrained Schemes</td>
      <td>Paradox Interactive,Paradox Interactive</td>
      <td>Mechs,Strategy,Turn-Based,Turn-Based Tactics,S...</td>
      <td>Single-player,Multi-player,Online Multi-Player...</td>
      <td>English,French,German,Russian</td>
      <td>128.0</td>
      <td>Action,Adventure,Strategy</td>
      <td>About This Game  From original BATTLETECH/Mec...</td>
      <td>NaN</td>
      <td>Minimum:,Requires a 64-bit processor and opera...</td>
      <td>Recommended:,Requires a 64-bit processor and o...</td>
      <td>$39.99</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Step One: Filtering the price between \\$19.99 and \\$59.99
The original price column will be the column we intend to filter. We have a problem to sort out before we proceed with our filtering. We cannot sort the original price column because it is not considered a numerical type. We can fix this by first converting the column to a character type, then remove the dollar sign through character string slicing. After we remove the dollar sign, we can convert the column to a numerical type. Now we can proceed with applying the filter. The total number of unique games in the dataset is now 4,338.


```python
Games.original_price
```




    0        $19.99
    1        $29.99
    2        $39.99
    3        $44.99
    4          Free
              ...  
    40828     $2.99
    40829     $2.99
    40830     $7.99
    40831     $9.99
    40832     $4.99
    Name: original_price, Length: 40833, dtype: object




```python
Games['original_price'] = Games['original_price'].str[1:]
```


```python
Games['original_price'] = pd.to_numeric(Games['original_price'],errors='coerce')
```


```python
Games = Games[(Games['original_price'] >= 19.99) & (Games['original_price'] <= 59.99)]
```


```python
Games.shape
```




    (4338, 20)



### Step Two: Choosing columns to use in the recommendation system
When choosing which columns to put in the recommendation system, we should be mindful of the characteristics gamer's value. The developer variable is important to include since developers often have the same team working on different games. Therefore each game produced by the same developer will have a similar style of gameplay. Genre variable provides a broad grouping of games with similarities in form, style, or subject matter. Popular Tags variable is an in-depth description of different gaming characteristics. The Game Details variable lists a game's online offering such as whether a game is single-player or multiplayer. The last variable would be the name of the game, which is valuable because sequels and prequels will be included in the recommendation.


```python
Games.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>types</th>
      <th>name</th>
      <th>desc_snippet</th>
      <th>recent_reviews</th>
      <th>all_reviews</th>
      <th>release_date</th>
      <th>developer</th>
      <th>publisher</th>
      <th>popular_tags</th>
      <th>game_details</th>
      <th>languages</th>
      <th>achievements</th>
      <th>genre</th>
      <th>game_description</th>
      <th>mature_content</th>
      <th>minimum_requirements</th>
      <th>recommended_requirements</th>
      <th>original_price</th>
      <th>discount_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://store.steampowered.com/app/379720/DOOM/</td>
      <td>app</td>
      <td>DOOM</td>
      <td>Now includes all three premium DLC packs (Unto...</td>
      <td>Very Positive,(554),- 89% of the 554 user revi...</td>
      <td>Very Positive,(42,550),- 92% of the 42,550 use...</td>
      <td>May 12, 2016</td>
      <td>id Software</td>
      <td>Bethesda Softworks,Bethesda Softworks</td>
      <td>FPS,Gore,Action,Demons,Shooter,First-Person,Gr...</td>
      <td>Single-player,Multi-player,Co-op,Steam Achieve...</td>
      <td>English,French,Italian,German,Spanish - Spain,...</td>
      <td>54.0</td>
      <td>Action</td>
      <td>About This Game Developed by id software, the...</td>
      <td>NaN</td>
      <td>Minimum:,OS:,Windows 7/8.1/10 (64-bit versions...</td>
      <td>Recommended:,OS:,Windows 7/8.1/10 (64-bit vers...</td>
      <td>19.99</td>
      <td>$14.99</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://store.steampowered.com/app/578080/PLAY...</td>
      <td>app</td>
      <td>PLAYERUNKNOWN'S BATTLEGROUNDS</td>
      <td>PLAYERUNKNOWN'S BATTLEGROUNDS is a battle roya...</td>
      <td>Mixed,(6,214),- 49% of the 6,214 user reviews ...</td>
      <td>Mixed,(836,608),- 49% of the 836,608 user revi...</td>
      <td>Dec 21, 2017</td>
      <td>PUBG Corporation</td>
      <td>PUBG Corporation,PUBG Corporation</td>
      <td>Survival,Shooter,Multiplayer,Battle Royale,PvP...</td>
      <td>Multi-player,Online Multi-Player,Stats</td>
      <td>English,Korean,Simplified Chinese,French,Germa...</td>
      <td>37.0</td>
      <td>Action,Adventure,Massively Multiplayer</td>
      <td>About This Game  PLAYERUNKNOWN'S BATTLEGROUND...</td>
      <td>Mature Content Description  The developers de...</td>
      <td>Minimum:,Requires a 64-bit processor and opera...</td>
      <td>Recommended:,Requires a 64-bit processor and o...</td>
      <td>29.99</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://store.steampowered.com/app/637090/BATT...</td>
      <td>app</td>
      <td>BATTLETECH</td>
      <td>Take command of your own mercenary outfit of '...</td>
      <td>Mixed,(166),- 54% of the 166 user reviews in t...</td>
      <td>Mostly Positive,(7,030),- 71% of the 7,030 use...</td>
      <td>Apr 24, 2018</td>
      <td>Harebrained Schemes</td>
      <td>Paradox Interactive,Paradox Interactive</td>
      <td>Mechs,Strategy,Turn-Based,Turn-Based Tactics,S...</td>
      <td>Single-player,Multi-player,Online Multi-Player...</td>
      <td>English,French,German,Russian</td>
      <td>128.0</td>
      <td>Action,Adventure,Strategy</td>
      <td>About This Game  From original BATTLETECH/Mec...</td>
      <td>NaN</td>
      <td>Minimum:,Requires a 64-bit processor and opera...</td>
      <td>Recommended:,Requires a 64-bit processor and o...</td>
      <td>39.99</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
Games = Games[['genre','game_details','popular_tags','developer','name']]
```

### Step Three: Drop all rows with null values
Usually, the first step in any project would be to eliminate null values. However, it is important to wait to perform this step. We have previously consolidated columns to only useful columns for the recommendation system. Now that the dataset only has useful columns, we can eliminate only rows where null values are present in the columns we have chosen. After eliminating null values the total unique games in the dataset are 3,999. We will also be adding a new column labeled Game_ID, which provides a numerical unique value to each game. 


```python
Games.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genre</th>
      <th>game_details</th>
      <th>popular_tags</th>
      <th>developer</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Action</td>
      <td>Single-player,Multi-player,Co-op,Steam Achieve...</td>
      <td>FPS,Gore,Action,Demons,Shooter,First-Person,Gr...</td>
      <td>id Software</td>
      <td>DOOM</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Action,Adventure,Massively Multiplayer</td>
      <td>Multi-player,Online Multi-Player,Stats</td>
      <td>Survival,Shooter,Multiplayer,Battle Royale,PvP...</td>
      <td>PUBG Corporation</td>
      <td>PLAYERUNKNOWN'S BATTLEGROUNDS</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Action,Adventure,Strategy</td>
      <td>Single-player,Multi-player,Online Multi-Player...</td>
      <td>Mechs,Strategy,Turn-Based,Turn-Based Tactics,S...</td>
      <td>Harebrained Schemes</td>
      <td>BATTLETECH</td>
    </tr>
  </tbody>
</table>
</div>




```python
Games.dropna(inplace = True)
```


```python
Games.shape
```




    (3999, 5)




```python
Games['Game_ID'] = range(0,3999)
```


```python
Games.isnull().values.any()
```




    False




```python
Games = Games.reset_index()
```

### Step Four: Combine selected column's values into string
Our next step is going to be creating a function that compiles all data in each column selected into one giant string. In order to do so, we are going to make an empty list called important features and then append the values of the desired columns. Then we create a column called important features, where we call the function on the dataset.


```python
def get_important_features(data):
    important_features = []
    for i in range(0, data.shape[0]):
        important_features.append(data['name'][i]+' '+data['developer'][i]+' '+data['popular_tags'][i]+' '+data['genre'][i]+data['game_details'][i])
        
    return important_features
```


```python
Games['important_features'] = get_important_features(Games)
Games.important_features.head(3)
```




    0    DOOM id Software FPS,Gore,Action,Demons,Shoote...
    1    PLAYERUNKNOWN'S BATTLEGROUNDS PUBG Corporation...
    2    BATTLETECH Harebrained Schemes Mechs,Strategy,...
    Name: important_features, dtype: object



### Step Five: Assemble similarity matrix
First, we will be using the count vectorizer function to transform a given text into a vector. The matrix consists of a frequency of words in a string. For example the string 'Action, Action, Adventure', the matrix will display a table with the word, Action, and a frequency of two. Then we can use the cosine similarity function to measure the correlation among the different games. This function produces a matrix with the correlations between each game. The matrix contains a numerical value from 0 to 1, where a variable closer to 1 is considered a good recommendation, and a variable closer to 0 is considered a poor recommendation. The diagonal line of the value 1 showcases a perfect correlation because it is the same game on each axis.


```python
cm = CountVectorizer().fit_transform(Games['important_features'])
```


```python
cs = cosine_similarity(cm)
```


```python
print(cs)
```

    [[1.         0.40406102 0.44932255 ... 0.4276686  0.18002057 0.19738551]
     [0.40406102 1.         0.34163336 ... 0.41871789 0.31520362 0.26363719]
     [0.44932255 0.34163336 1.         ... 0.26702293 0.27136386 0.33377867]
     ...
     [0.4276686  0.41871789 0.26702293 ... 1.         0.35533453 0.27272727]
     [0.18002057 0.31520362 0.27136386 ... 0.35533453 1.         0.07106691]
     [0.19738551 0.26363719 0.33377867 ... 0.27272727 0.07106691 1.        ]]


### Step Six: Use the Recommendation System
Our last step would be to enter the name of the game we wish to get recommendations from. In this case, I have chosen the game Doom Eternal. We then create a new object called title_id, where we obtain the Game_ID value for Doom Eternal, which we assigned to each title in Step 3. After this step, we are going to create a list of enumerations that contain the similarity score between each game and Doom Eternal. Then we sort the similarity score in descending order to receive the games with the highest similarities to Doom Eternal. I have chosen to display the top 7 games that are recommended to us based on the characteristics of Doom Eternal.


```python
title = 'DOOM Eternal'
title_id = Games[Games.name == title]['Game_ID'].values[0]
```


```python
scores = list(enumerate(cs[title_id]))
```


```python
sorted_scores = sorted(scores, key = lambda x:x[1], reverse = True)
sorted_scores = sorted_scores[1:]
```


```python
j = 0
print('The 7 most recommended games to', title, 'are:\n')
for item in sorted_scores:
    game_title = Games[Games.Game_ID == item[0]]['name'].values[0]
    print(j+1, game_title)
    j = j+1
    if j > 6:
        break
```

    The 7 most recommended games to DOOM Eternal are:
    
    1 Doom 3: BFG Edition
    2 DOOM
    3 Dead Space™ 2
    4 DUSK
    5 Max Payne 3
    6 Unreal Tournament 3 Black
    7 Crysis 2 - Maximum Edition


### Conclusion

When observing the top seven results we can see the similarities between the games. The more similarities in each column the higher the ranking will be. For instance, Doom 3: BFG Edition and DOOM have similarities in every column. While the bottom four recommendations have values in common in the genre, game details, and popular tags columns. From my personal experience playing five out of the seven recommended games, I would like to have these games recommended to me based on my interest of DOOM Eternal.


```python
Games = Games.set_index('name')
```


```python
Games.loc[['DOOM Eternal','Doom 3: BFG Edition','DOOM','Dead Space™ 2','DUSK','Max Payne 3','Unreal Tournament 3 Black','Crysis 2 - Maximum Edition'],
         ['genre','game_details','popular_tags','developer']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genre</th>
      <th>game_details</th>
      <th>popular_tags</th>
      <th>developer</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>DOOM Eternal</th>
      <td>Action</td>
      <td>Single-player,Multi-player,Online Multi-Player...</td>
      <td>Gore,Violent,Action,FPS,Great Soundtrack,Demon...</td>
      <td>id Software</td>
    </tr>
    <tr>
      <th>Doom 3: BFG Edition</th>
      <td>Action</td>
      <td>Single-player,Multi-player,Steam Achievements,...</td>
      <td>FPS,Horror,Action,Shooter,Classic,Sci-fi,Singl...</td>
      <td>id Software</td>
    </tr>
    <tr>
      <th>DOOM</th>
      <td>Action</td>
      <td>Single-player,Multi-player,Co-op,Steam Achieve...</td>
      <td>FPS,Gore,Action,Demons,Shooter,First-Person,Gr...</td>
      <td>id Software</td>
    </tr>
    <tr>
      <th>Dead Space™ 2</th>
      <td>Action</td>
      <td>Single-player,Multi-player,Partial Controller ...</td>
      <td>Horror,Action,Sci-fi,Space,Third Person,Surviv...</td>
      <td>Visceral Games</td>
    </tr>
    <tr>
      <th>DUSK</th>
      <td>Action,Indie</td>
      <td>Single-player,Online Multi-Player,Steam Achiev...</td>
      <td>FPS,Retro,Action,Fast-Paced,Great Soundtrack,H...</td>
      <td>David Szymanski</td>
    </tr>
    <tr>
      <th>Max Payne 3</th>
      <td>Action</td>
      <td>Single-player,Multi-player,Steam Achievements,...</td>
      <td>Action,Third-Person Shooter,Bullet Time,Story ...</td>
      <td>Rockstar Studios</td>
    </tr>
    <tr>
      <th>Unreal Tournament 3 Black</th>
      <td>Action</td>
      <td>Single-player,Multi-player,Co-op,Steam Achieve...</td>
      <td>FPS,Action,Multiplayer,Arena Shooter,Shooter,S...</td>
      <td>Epic Games, Inc.</td>
    </tr>
    <tr>
      <th>Crysis 2 - Maximum Edition</th>
      <td>Action</td>
      <td>Single-player,Partial Controller Support</td>
      <td>Action,FPS,Sci-fi,Shooter,Singleplayer,Multipl...</td>
      <td>Crytek Studios</td>
    </tr>
  </tbody>
</table>
</div>


