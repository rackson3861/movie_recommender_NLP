"""                          @Author : Rackson  
        Indian Institute of Technology (Indian School Of Mines) Dhanbad

***************************MOVIE RECOMMENDATION*****************************

"""   

""" Control the number of images for comparison by changing the no_of_images variable"""
""" Control the number of suggestions by changing the no_of_images variable"""
no_of_images = 15000
no_of_recommendations = 10
# importing the relevant libraries

import pandas as pd
import numpy as np

# reading the bag_of_words.csv file
words = pd.read_csv('bag_of_words.csv',header = -1)

# Due to lack of computational power using smaller number of images.
words = words.iloc[0:no_of_images,:]

# preprocessing the bag of words so as to reduce paramaters

# Uncomment  line 34 and  35 to use stemming

# importing relevant libraries
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#desc = re.sub('[^a-zA-Z]',' ',words[1][0])   #first column then row number
corpus = []
for i in range(no_of_images):
    desc = re.sub('[^a-zA-Z]',' ',words[1][i])
    desc = words.iloc[i,1].lower()
    desc = desc.split()
    #ps = PorterStemmer()
    #desc = [ps.stem(word) for word in desc if not word in set(stopwords.words('english'))]
    desc = ' '.join(desc)
    corpus.append(desc)
    print(i)

# importing CountVectorizer 
from sklearn.feature_extraction.text import CountVectorizer

# Converting corpus to a usable dataframe

df = pd.DataFrame(corpus)

# Using CountVectorizer to make a sparse matrix of each word of bag of words 

count  = CountVectorizer()
df = df[0]
count_matrix = count.fit_transform(df)
print(count_matrix)
print(count_matrix.shape)

# Using cosine similarity to compute the similarity confidence score between each movies

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(count_matrix,count_matrix)

#Uncomment below two lines  63 and 64 for and comment 57 and 59 to use linear kernel

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
# finding indices of every title
indices = pd.Series(words.index, index=titles)

#-------Recommendation for the movie which client have watched recently---------
movie_user_likes = "The Men"
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
