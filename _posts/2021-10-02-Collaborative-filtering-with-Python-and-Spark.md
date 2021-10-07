---
title: "Collaborative Filtering with Python and Spark"
excerpt: "Collaborative Filtering with Python and Spark"

header:
  image: "../assets/images/posts/2021-10-02-Collaborative-filtering-with-Python-and-Spark/Data-center1.jpg"
  teaser: "../assets/images/posts/2021-10-02-Collaborative-filtering-with-Python-and-Spark/Data-center1.jpg"
  caption: "Cloud is the digital wonderland of Internet of Things, powered by Artificial Intelligence and Big Data - Enamul Haque"
  
---

Recommendation systems are a collection of algorithms used to recommend items to users based on information taken from the user. These systems have become ubiquitous can be commonly seen in online stores, movies databases and job finders.

<h4>Table of contents</h4>
<div class="alert alert-block alert-info" style="margin-top: 20px">
    <ol>
        <li><a href="#ref1">Collaborative Filtering with Python</a></li>
        <li><a href="#ref2">Collaborative Filtering with Pyspark</a></li>
    </ol>
</div>
<br>

The installation of **Python** and **Pyspark**  and the introduction of **theory**  the Recommendation systems is given [here.](./Machine-Learning-with-Python-and-Spark)

<h2 id="understanding_data">Understanding the Data</h2>

To acquire and extract the data, 
Dataset acquired from [GroupLens](http://grouplens.org/datasets/movielens/). Lets download the dataset.


```python
import requests
import zipfile
```




```python
url = r'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/moviedataset.zip'
output ='moviedataset.zip'
r = requests.get(url)
with open(output, 'wb') as f:
    f.write(r.content)
with zipfile.ZipFile(output) as item:
    item.extractall()
```




```python
#Dataframe manipulation library
import pandas as pd
#Math functions, we'll only need the sqrt function so let's import only that
from math import sqrt
import numpy as np
```




```python
#Storing the movie information into a pandas dataframe
movies_df = pd.read_csv('./ml-latest/movies.csv')
#Storing the user information into a pandas dataframe
ratings_df = pd.read_csv('./ml-latest/ratings.csv')
```




```python
#Head is a function that gets the first N rows of a dataframe. N's default is 5.
movies_df.head(1)
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
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
  </tbody>
</table>

</div>



Let's remove the year from the title column and place it into its own one by using the handy . 


```python
#Using regular expressions to find a year stored between parentheses
#We specify the parantheses so we don't conflict with movies that have years in their titles
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
#Removing the parentheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
#Removing the years from the 'title' column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
#Applying the strip function to get rid of any ending whitespace characters that may have appeared
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
```




```python
movies_df.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {    vertical-align: top;}.dataframe thead th {    text-align: right;}

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>1995</td>
    </tr>
  </tbody>
</table>

</div>




```python
#Dropping the genres column
movies_df = movies_df.drop('genres', 1)
```




```python
movies_df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {    vertical-align: top;}.dataframe thead th {    text-align: right;}

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>1995</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji</td>
      <td>1995</td>
    </tr>
  </tbody>
</table>

</div>




```python
ratings_df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {    vertical-align: top;}.dataframe thead th {    text-align: right;}

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>169</td>
      <td>2.5</td>
      <td>1204927694</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2471</td>
      <td>3.0</td>
      <td>1204927438</td>
    </tr>
  </tbody>
</table>

</div>



Every row in the ratings dataframe has a user id associated with at least one movie, a rating and a timestamp showing when they reviewed it. We won't be needing the timestamp column, so let's drop it to save on memory.


```python
#Drop removes a specified row or column from a dataframe
ratings_df = ratings_df.drop('timestamp', 1)
```


```python
ratings_df.to_csv("ratings.csv", encoding='utf-8', index=False)
```

The process for creating a User Based recommendation system is as follows:

- Select a user with the movies the user has watched
- Based on his rating to movies, find the top X neighbours 
- Get the watched movie record of the user for each neighbour.
- Calculate a similarity score using some formula
- Recommend the items with the highest score


Let's begin by creating an input user to recommend movies to:

Notice: To add more movies, simply increase the amount of elements in the userInput. 

<a id="ref1"></a>

##  Collaborative Filtering with Python


```python
#Dataframe manipulation library
import pandas as pd
#Math functions, we'll only need the sqrt function so let's import only thatfrom math 
import sqrtimport numpy as np
import matplotlib.pyplot as plt%matplotlib inline
```

The first technique we're going to take a look at is called __Collaborative Filtering__, which is also known as __User-User Filtering__. 


It attempts to find users that have similar preferences and opinions as the input and then recommends items that they have liked to the input.


The process for creating a User Based recommendation system is as follows:

- Select a user with the movies the user has watched
- Based on his rating to movies, find the top X neighbours 
- Get the watched movie record of the user for each neighbour.
- Calculate a similarity score using some formula
- Recommend the items with the highest score






```python
userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ] 
inputMovies = pd.DataFrame(userInput)
inputMovies
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {    vertical-align: top;}.dataframe thead th {    text-align: right;}

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Breakfast Club, The</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Toy Story</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jumanji</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Pulp Fiction</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Akira</td>
      <td>4.5</td>
    </tr>
  </tbody>
</table>

</div>



#### Add movieId to input user

With the input complete, let's extract the input movies's ID's from the movies dataframe and add them into it.

We can achieve this by first filtering out the rows that contain the input movies' title and then merging this subset with the input dataframe. We also drop unnecessary columns for the input to save memory space.


```python
#Filtering out the movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]

```


```python
#Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
```


```python
#Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('year', 1)
```


```python
#Final input dataframe
#If a movie you added in above isn't here, then it might not be in the original 
#dataframe or it might spelled differently, please check capitalisation.
inputMovies
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {    vertical-align: top;}.dataframe thead th {    text-align: right;}

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>296</td>
      <td>Pulp Fiction</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1274</td>
      <td>Akira</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1968</td>
      <td>Breakfast Club, The</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>

</div>



#### The users who has seen the same movies

Now with the movie ID's in our input, we can now get the subset of users that have watched and reviewed the movies in our input.



```python
#Filtering out users that have watched movies that the input has watched and storing it
userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
userSubset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {    vertical-align: top;}.dataframe thead th {    text-align: right;}

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>4</td>
      <td>296</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>441</th>
      <td>12</td>
      <td>1968</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>479</th>
      <td>13</td>
      <td>2</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>531</th>
      <td>13</td>
      <td>1274</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>681</th>
      <td>14</td>
      <td>296</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>

</div>




```python
#Groupby creates several sub dataframes where they all have the same value in the column specified as the parameter
userSubsetGroup = userSubset.groupby(['userId'])
```

lets look at one of the users, e.g. the one with userID=1130


```python
userSubsetGroup.get_group(1130)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {    vertical-align: top;}.dataframe thead th {    text-align: right;}

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>104167</th>
      <td>1130</td>
      <td>1</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>104168</th>
      <td>1130</td>
      <td>2</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>104214</th>
      <td>1130</td>
      <td>296</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>104363</th>
      <td>1130</td>
      <td>1274</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>104443</th>
      <td>1130</td>
      <td>1968</td>
      <td>4.5</td>
    </tr>
  </tbody>
</table>

</div>




```python
#Sorting it so users with movie most in common with the input will have priority
userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)
```

Now lets look at the first user


```python
userSubsetGroup[0:3]
```




    [(75,
            userId  movieId  rating
      7507      75        1     5.0
      7508      75        2     3.5
      7540      75      296     5.0
      7633      75     1274     4.5
      7673      75     1968     5.0),
     (106,
            userId  movieId  rating
      9083     106        1     2.5
      9084     106        2     3.0
      9115     106      296     3.5
      9198     106     1274     3.0
      9238     106     1968     3.5),
     (686,
             userId  movieId  rating
      61336     686        1     4.0
      61337     686        2     3.0
      61377     686      296     4.0
      61478     686     1274     4.0
      61569     686     1968     5.0)]



We will select a subset of users to iterate through. This limit is imposed because we don't want to waste too much time going through every single user.


```python
userSubsetGroup = userSubsetGroup[0:100]
```

Next, we are going to compare all users to our specified user and find the one that is most similar.  
we're going to find out how similar each user is to the input through the __Pearson Correlation Coefficient__. It is used to measure the strength of a linear association between two variables.  Pearson correlation is invariant to scaling, i.e. multiplying all elements by a nonzero constant or adding any constant to all elements.  .

The values given by the formula vary from r = -1 to r = 1, where 1 forms a direct correlation between the two entities (it means a perfect positive correlation) and -1 forms a perfect negative correlation.  In our case, a 1 means that the two users have similar tastes while a -1 means the opposite.

Now, we calculate the Pearson Correlation between input user and subset group, and store it in a dictionary, where the key is the user Id and the value is the coefficient



```python
#Store the Pearson Correlation in a dictionary, where the key is the user Id and the value is the coefficient
pearsonCorrelationDict = {}

#For every user group in our subset
for name, group in userSubsetGroup:
    #Let's start by sorting the input and current user group so the values aren't mixed up later on
    group = group.sort_values(by='movieId')
    inputMovies = inputMovies.sort_values(by='movieId')
    #Get the N for the formula
    nRatings = len(group)
    #Get the review scores for the movies that they both have in common
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
    #And then store them in a temporary buffer variable in a list format to facilitate future calculations
    tempRatingList = temp_df['rating'].tolist()
    #Let's also put the current user group reviews in a list format
    tempGroupList = group['rating'].tolist()
    #Now let's calculate the pearson correlation between two users, so called, x and y
    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
    Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)
    
    #If the denominator is different than zero, then divide, else, 0 correlation.
    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
    else:
        pearsonCorrelationDict[name] = 0

```


```python
pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))
pearsonDF.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {    vertical-align: top;}.dataframe thead th {    text-align: right;}

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>similarityIndex</th>
      <th>userId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.827278</td>
      <td>75</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.586009</td>
      <td>106</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.832050</td>
      <td>686</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.576557</td>
      <td>815</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.943456</td>
      <td>1040</td>
    </tr>
  </tbody>
</table>

</div>



#### The top x similar users to input user

Now let's get the top 50 users that are most similar to the input.


```python
topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
topUsers.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {    vertical-align: top;}.dataframe thead th {    text-align: right;}

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>similarityIndex</th>
      <th>userId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>64</th>
      <td>0.961678</td>
      <td>12325</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0.961538</td>
      <td>6207</td>
    </tr>
    <tr>
      <th>55</th>
      <td>0.961538</td>
      <td>10707</td>
    </tr>
    <tr>
      <th>67</th>
      <td>0.960769</td>
      <td>13053</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.943456</td>
      <td>1040</td>
    </tr>
  </tbody>
</table>

</div>




#### Rating of selected users to all movies

We're going to do this by taking the weighted average of the ratings of the movies using the Pearson Correlation as the weight. But to do this, we first need to get the movies watched by the users in our __pearsonDF__ from the ratings dataframe and then store their correlation in a new column called _similarityIndex". This is achieved below by merging of these two tables.


```python
topUsersRating=topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')topUsersRating.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {    vertical-align: top;}.dataframe thead th {    text-align: right;}

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>similarityIndex</th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.961678</td>
      <td>12325</td>
      <td>1</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.961678</td>
      <td>12325</td>
      <td>2</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.961678</td>
      <td>12325</td>
      <td>3</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.961678</td>
      <td>12325</td>
      <td>5</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.961678</td>
      <td>12325</td>
      <td>6</td>
      <td>2.5</td>
    </tr>
  </tbody>
</table>

</div>



Now all we need to do is simply multiply the movie rating by its weight (The similarity index), then sum up the new ratings and divide it by the sum of the weights.

We can easily do this by simply multiplying two columns, then grouping up the dataframe by movieId and then dividing two columns:

It shows the idea of all similar users to candidate movies for the input user:


```python
#Multiplies the similarity by the user's ratings
topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
topUsersRating.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {    vertical-align: top;}.dataframe thead th {    text-align: right;}

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>similarityIndex</th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>weightedRating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.961678</td>
      <td>12325</td>
      <td>1</td>
      <td>3.5</td>
      <td>3.365874</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.961678</td>
      <td>12325</td>
      <td>2</td>
      <td>1.5</td>
      <td>1.442517</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.961678</td>
      <td>12325</td>
      <td>3</td>
      <td>3.0</td>
      <td>2.885035</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.961678</td>
      <td>12325</td>
      <td>5</td>
      <td>0.5</td>
      <td>0.480839</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.961678</td>
      <td>12325</td>
      <td>6</td>
      <td>2.5</td>
      <td>2.404196</td>
    </tr>
  </tbody>
</table>

</div>




```python
#Applies a sum to the topUsers after grouping it up by userId
tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
tempTopUsersRating.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {    vertical-align: top;}.dataframe thead th {    text-align: right;}

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sum_similarityIndex</th>
      <th>sum_weightedRating</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>38.376281</td>
      <td>140.800834</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38.376281</td>
      <td>96.656745</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.253981</td>
      <td>27.254477</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.929294</td>
      <td>2.787882</td>
    </tr>
    <tr>
      <th>5</th>
      <td>11.723262</td>
      <td>27.151751</td>
    </tr>
  </tbody>
</table>

</div>




```python
#Creates an empty dataframe
recommendation_df = pd.DataFrame()
#Now we take the weighted average
recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
recommendation_df['movieId'] = tempTopUsersRating.index
recommendation_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {    vertical-align: top;}.dataframe thead th {    text-align: right;}

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weighted average recommendation score</th>
      <th>movieId</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>3.668955</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.518658</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.657941</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.000000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.316058</td>
      <td>5</td>
    </tr>
  </tbody>
</table>

</div>



Now let's sort it and see the top 20 movies that the algorithm recommended!


```python
recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)recommendation_df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {    vertical-align: top;}.dataframe thead th {    text-align: right;}

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weighted average recommendation score</th>
      <th>movieId</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5073</th>
      <td>5.0</td>
      <td>5073</td>
    </tr>
    <tr>
      <th>3329</th>
      <td>5.0</td>
      <td>3329</td>
    </tr>
    <tr>
      <th>2284</th>
      <td>5.0</td>
      <td>2284</td>
    </tr>
    <tr>
      <th>26801</th>
      <td>5.0</td>
      <td>26801</td>
    </tr>
    <tr>
      <th>6776</th>
      <td>5.0</td>
      <td>6776</td>
    </tr>
    <tr>
      <th>6672</th>
      <td>5.0</td>
      <td>6672</td>
    </tr>
    <tr>
      <th>3759</th>
      <td>5.0</td>
      <td>3759</td>
    </tr>
    <tr>
      <th>3769</th>
      <td>5.0</td>
      <td>3769</td>
    </tr>
    <tr>
      <th>3775</th>
      <td>5.0</td>
      <td>3775</td>
    </tr>
    <tr>
      <th>90531</th>
      <td>5.0</td>
      <td>90531</td>
    </tr>
  </tbody>
</table>

</div>




```python
movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {    vertical-align: top;}.dataframe thead th {    text-align: right;}

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2200</th>
      <td>2284</td>
      <td>Bandit Queen</td>
      <td>1994</td>
    </tr>
    <tr>
      <th>3243</th>
      <td>3329</td>
      <td>Year My Voice Broke, The</td>
      <td>1987</td>
    </tr>
    <tr>
      <th>3669</th>
      <td>3759</td>
      <td>Fun and Fancy Free</td>
      <td>1947</td>
    </tr>
    <tr>
      <th>3679</th>
      <td>3769</td>
      <td>Thunderbolt and Lightfoot</td>
      <td>1974</td>
    </tr>
    <tr>
      <th>3685</th>
      <td>3775</td>
      <td>Make Mine Music</td>
      <td>1946</td>
    </tr>
    <tr>
      <th>4978</th>
      <td>5073</td>
      <td>Son's Room, The (Stanza del figlio, La)</td>
      <td>2001</td>
    </tr>
    <tr>
      <th>6563</th>
      <td>6672</td>
      <td>War Photographer</td>
      <td>2001</td>
    </tr>
    <tr>
      <th>6667</th>
      <td>6776</td>
      <td>Lagaan: Once Upon a Time in India</td>
      <td>2001</td>
    </tr>
    <tr>
      <th>9064</th>
      <td>26801</td>
      <td>Dragon Inn (Sun lung moon hak chan)</td>
      <td>1992</td>
    </tr>
    <tr>
      <th>18106</th>
      <td>90531</td>
      <td>Shame</td>
      <td>2011</td>
    </tr>
  </tbody>
</table>

</div>



<a id="ref2"></a>

### Collaborative Filtering with Pyspark

First thing to do is start a Spark Session


```python
import findspark
```


```python
findspark.init()
```


```python
from pyspark.sql import SparkSession
```

If your computer has less than 5g of RAM change the memory of spark


```python
spark = SparkSession.builder \
    .master('local[*]') \
    .config("spark.driver.memory", "5g") \
    .appName('rec') \
    .getOrCreate()
```


```python
from pyspark.ml.evaluation import RegressionEvaluatorfrom pyspark.ml.recommendation import ALS
```


```python
data = spark.read.csv('ratings.csv',inferSchema=True,header=True)
```


```python
data.head()
```




    Row(userId=1, movieId=169, rating=2.5)




```python
data.describe().show()
```

    +-------+------------------+------------------+------------------+
    |summary|            userId|           movieId|            rating|
    +-------+------------------+------------------+------------------+
    |  count|          22884377|          22884377|          22884377|
    |   mean|123545.22803517876|11408.161728851084|3.5260770044122243|
    | stddev|  71474.6902962076| 24136.87588274057|1.0611734340135037|
    |    min|                 1|                 1|               0.5|
    |    max|            247753|            151711|               5.0|
    +-------+------------------+------------------+------------------+


​    


```python
# Smaller dataset so we will use 0.8 / 0.2
(training, test) = data.randomSplit([0.8, 0.2])
```


```python
# Build the recommendation model using ALS on the training data
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating")
```

Attention: we requiere 5g of free  RAM memory


```python
model = als.fit(training)
```

Now let's see how the model performed!


```python
# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
```


```python
predictions.show()

```

```
+------+-------+------+----------+
|userId|movieId|rating|prediction|
+------+-------+------+----------+
|178254|    148|   3.0| 4.6890635|
| 90446|    148|   2.0| 2.8395023|
|134189|    148|   5.0| 3.7372081|
|158304|    148|   3.0| 3.2317648|
|236731|    148|   3.0| 2.6928616|
|233017|    148|   2.0| 2.8125327|
|119850|    148|   4.0| 3.2640007|
|219880|    148|   5.0|  5.158979|
|108678|    148|   4.0|  2.820398|
| 48620|    148|   2.0|  3.032034|
|112136|    148|   2.0| 3.3153908|
|147802|    148|   5.0|  3.659751|
| 33400|    148|   3.0| 3.9928482|
| 88163|    148|   3.0| 2.1727314|
|191207|    148|   3.0| 3.5695121|
| 37586|    148|   3.0|0.97570467|
|142515|    148|   3.0| 2.9032598|
|184545|    148|   1.0| 2.0106587|
|169266|    148|   1.0|0.44079798|
| 86384|    148|   3.0|  3.259998|
+------+-------+------+----------+
only showing top 20 rows

```


​    

So now that we have the model, how would you actually supply a recommendation to a user?

The same way we did with the test data! For example:


```python
single_user = test.filter(test['userId']==11).select(['movieId','userId'])
```


```python
# User had 10 ratings in the test data set 
single_user.show()
```

    +-------+------+
    |movieId|userId|
    +-------+------+
    |      3|    11|
    |    186|    11|
    |    908|    11|
    |   1220|    11|
    |   1225|    11|
    |   1266|    11|
    |   1275|    11|
    |   2599|    11|
    |   2641|    11|
    |   2687|    11|
    |   2993|    11|
    |   3686|    11|
    |   3826|    11|
    +-------+------+


​    


```python
reccomendations = model.transform(single_user)
```


```python
reccomendations.orderBy('prediction',ascending=False).show()
```

    +-------+------+----------+
    |movieId|userId|prediction|
    +-------+------+----------+
    |   1275|    11| 3.4204795|
    |   1225|    11| 3.4091446|
    |   2687|    11| 3.3668394|
    |   1266|    11|  3.320037|
    |    908|    11| 3.2962897|
    |   3686|    11|  3.268262|
    |   1220|    11|   3.22914|
    |   2993|    11|    3.0513|
    |   2599|    11| 3.0509036|
    |   2641|    11| 2.8271532|
    |      3|    11| 2.8164563|
    |    186|    11| 2.8024044|
    |   3826|    11| 2.7049482|
    +-------+------+----------+

 





You can download the notebook [here](https://github.com/ruslanmv/Machine-Learning-with-Python-and-Spark/blob/master/Recommender-Systems/Recommender-Systems.ipynb)

**Congratulations!** We have practiced Collaborative Filtering with Python and Spark

