"""                          @Author : Rackson  
        Indian Institute of Technology (Indian School Of Mines) Dhanbad

***************************MOVIE RECOMMENDATION*****************************

"""

# Preparation of Bag of Words 


# importing the libraries
import pandas as pd
import numpy as np

# dataframe of metadata dataset.
metadata = pd.read_csv('movies_metadata.csv',header = 0)

# taking dataframes of different relevant features
overview = metadata['overview']
language = metadata['original_language']
tag_line = metadata['tagline']
adult = metadata['adult']

# removing two faulty entries of adult and making them true by default

print(adult.unique())

for i in range(45466):
    if adult[i]=='FALSE' or adult[i]=='TRUE':
        continue;
    else:
        adult[i]='TRUE'
    print(i)

# verifying whether faulty inputs are removed or not
print(adult.unique())

# reading the belongs_to_collection dataframe
# Note: the collection substring in each name of collections has already been removed in csv file.
collection = pd.read_csv('collection.csv',header = 0)

collection = collection['names']

# Removing white spaces between the genre names such as 'Science Fiction' to 'ScienceFiction'
# because Science and Fiction represent same entities rather than different
for i in range(45464):
    collection[i] = str(collection[i]).replace("Collection","")
genre = pd.read_csv('genre.csv',header = -1)
genre = genre.iloc[:,0:8]
for i in range(45464):
    for j in range(8):
        genre.iloc[i][j] = str(genre.iloc[i][j]).replace(' ','')
    print(i)
    
genre_concat = genre.iloc[:,0]
# separating the genres and concatenating them for each moveis
for i in range(45464):
    for j in range(8):
        if(j>0):
            if(str(genre.iloc[i][j]) =='nan'):
                continue
            else:
                genre_concat[i] = str(genre_concat.iloc[i]) + " " + str(genre.iloc[i][j])
            print(i)

# reading the countries section as dataframe and removing 
# faulty entries (nan values to to empty string '')

countries = pd.read_csv('countries.csv',header = -1)

cuntry = countries[3]

print(cuntry.unique())


for i in range(45466):
    if str(cuntry[i]) == "nan":
        cuntry[i] = '';

for i in range(45466):
    for j in range(25):
        if 9*j+9+3<225:
            if str(countries.iloc[i,9*j+9+3]) == "nan":
                continue
            else:
                cuntry[i] = cuntry[i] + " " + str(countries.iloc[i,9*j+9+3]) 
        print(i)
        
companies = pd.read_csv('companies.csv',header = -1)
compy = companies[3]

for i in range(45466):
    if str(compy[i]) == "nan":
        compy[i] = '';
    print(i)

for i in range(45466):
    for j in range(26):
        if 8*j+8+3<208:
            if str(companies.iloc[i,8*j+8+3]) == "nan":
                continue
            else:
                compy[i] = compy[i] + " " + str(companies.iloc[i,8*j+8+3]) 
        print(i)
        
# renaming the columns for each separated features for the purpose of concatenation of feature columns
adult = adult.rename(0)
collection = collection.rename(1)
language = language.rename(2)
overview = overview.rename(3)
tag_line  = tag_line.rename(4)
compy = compy.rename(5)
cuntry = cuntry.rename(6)
genre_concat = genre_concat.rename(7)

adult = adult.iloc[0:45464]
collection = collection.iloc[0:45464]
language = language.iloc[0:45464]
overview = overview.iloc[0:45464]
tag_line = tag_line.iloc[0:45464]
compy = compy.iloc[0:45464]
cuntry = cuntry.iloc[0:45464]

#concatenating the different feature columns
pf = pd.concat([adult,collection,language,overview,tag_line,compy,cuntry,genre_concat],axis = 1)

df = pf[0]
 # making the bag of words column by concatenating the different feature values.
for i in range(45464):
    for j in range(8):
        if j>0:
            if(str(pf.iloc[i,j])=="nan"):
                continue
            df[i] = df[i] + " " + str(pf.iloc[i,j])

#concatenating the bag of words model column entries to there respective movie title values.

bag_words = metadata['original_title']

bag_of_words = pd.concat([bag_words,df],axis = 1)

# Saving the bag_of_words.csv file.

bag_of_words.to_csv('bag_of_words.csv')

