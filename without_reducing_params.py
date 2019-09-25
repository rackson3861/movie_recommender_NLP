"""                          @Author : Rackson  
        Indian Institute of Technology (Indian School Of Mines) Dhanbad

***************************MOVIE RECOMMENDATION*****************************

"""   

""" Control the number of images for comparison by changing the no_of_images variable"""
""" Control the number of suggestions by changing the no_of_images variable"""
no_of_images = 1500
no_of_recommendations = 10
# importing the relevant libraries

import pandas as pd
import numpy as np

# reading the bag_of_words.csv file
words = pd.read_csv('bag_of_words.csv',header = -1)

# Due to lack of computational power using smaller number of images.
words = words.iloc[0:no_of_images,:]

# importing CountVectorizer 

from sklearn.feature_extraction.text import CountVectorizer

# Getting the bag of words column from complete dataframe of words

df = words[1]

# Using CountVectorizer to make a sparse matrix of each word of bag of words 

count  = CountVectorizer()


count_matrix = count.fit_transform(df)
print(count_matrix)
print(count_matrix.shape)
from sklearn.metrics.pairwise import cosine_similarity

# Using cosine similarity to compute the similarity confidence score between each movies


cosine_sim = cosine_similarity(count_matrix,count_matrix)
count_matrix = pd.DataFrame(count_matrix)

#Uncomment below two lines  49 and 50 for and comment 44 and 45 to use linear kernel

#from sklearn.metrics.pairwise import linear_kernel
#cosine_sim = linear_kernel(count_matrix,count_matrix)

# renaming the columns of main dataframe of words
words.columns = ['title','description']
#---Function to get title and Index of movies for recommendation

def get_title_from_index(index):
    return words[words.index == index]["title"].values[0]
def get_index_from_title(title):
    return words[words.title == title]["index"].values[0]

words = words.reset_index()
titles = words['title']
# Finding indices of every title
indices = pd.Series(words.index, index=titles)

#-------Recommendation for the movie which client have watched recently---------
movie_user_likes = "Toy Story"
movie_index = get_index_from_title(movie_user_likes)
similar_movies = list(enumerate(cosine_sim[movie_index]))
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]

# Printing the top 10 relevant movies.
i=0
print("\n\n")
print("Hello world !!! Enjoy these movies in future --> \n ")
for element in sorted_similar_movies:
    print( str(i+1) + " -> " + str(get_title_from_index(element[0])))
    i=i+1
    if i>no_of_recommendations-1:
        break