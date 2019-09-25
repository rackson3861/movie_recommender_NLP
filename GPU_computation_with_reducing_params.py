"""                          @Author : Rackson  
        Indian Institute of Technology (Indian School Of Mines) Dhanbad

***************************MOVIE RECOMMENDATION*****************************

"""
""" Control the number of images for comparison by changing the no_of_images variable"""

no_of_images = 1500

# importing relevant libraries
import pandas as pd
import numpy as np

# reading the bag_of_words.csv file
words = pd.read_csv('bag_of_words.csv',header = -1)

# Due to lack of computational power using smaller number of images.

words = words.iloc[0:no_of_images,:]

# preprocessing the bag of words so as to reduce paramaters

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
    ps = PorterStemmer()
    desc = [ps.stem(word) for word in desc if not word in set(stopwords.words('english'))]
    desc = ' '.join(desc)
    corpus.append(desc)
    print(i)

# Converting corpus to a usable dataframe

df = pd.DataFrame(corpus)

# importing CountVectorizer 

from sklearn.feature_extraction.text import CountVectorizer

# Using CountVectorizer to make a sparse matrix of each word of bag of words 

count  = CountVectorizer()
df = df[0]
count_matrix = count.fit_transform(df)
print(count_matrix)
print(count_matrix.shape)
n = no_of_images
k = 1231
X = count_matrix
X = np.asarray(X)
print(X.shape)

# Parallel Computing using NVIDIA GEForce GTX 1050...4GB RAM
# Recommending top 3 movies 

from numba import cuda # Nvidia's GPU Library
import numpy as np 
import math #Doing mathematical calculations like dot product, Numpy not supported in GPU
import random

# Defining a kernel function                
@cuda.jit('void(float64[:,:],float64[:],int32[:],float64[:],int32[:],float64[:],int32[:])')
def cuda_dist(X,first_best_val,first_best_index,second_best_val,second_best_index,third_best_val,third_best_index):
    """This kernel function will be executed by a thread."""
    # Mapping of thread to row
    row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x;
    if ((row >= n)): # Acccept threads only within matrix boundaries
        return
    first_best_val[row] = SMALL # cosine similarity with first-closest item
    first_best_index[row] = -1 # index of first-closest item will be stored here
    second_best_val[row] = SMALL # cosine similarity with second-closest item
    second_best_index[row] = -1 # index of second-closest item will be stored here
    third_best_val[row] = SMALL # cosine similarity with third-closest item
    third_best_index[row] = -1 # index of third-closest item will be stored here
    for i in range(n):
        if(i==row): # Same item
            continue
        #Variables for Dot-product computation, as we can't use numpy operations within GPU
        tmp = 0.0
        magnitude1 = 0.0
        magnitude2 = 0.0
        for j in range(k):
            tmp += X[row,j] * X[i,j]
            magnitude1 += (X[row,j]* X[row,j])
            magnitude2 += (X[i,j]* X[i,j])
        tmp /= (math.sqrt(magnitude1)*math.sqrt(magnitude2)) #Dot product value Dot_product(a,b) = a.b/(|a|*|b|)
        if(tmp>=first_best_val[row]):
            third_best_val[row] = second_best_val[row]
            third_best_index[row] = second_best_index[row]
            second_best_val[row] = first_best_val[row]
            second_best_index[row] = first_best_index[row]
            first_best_val[row] = tmp
            first_best_index[row] = i
        elif(tmp>=second_best_val[row]):
            third_best_val[row] = second_best_val[row]
            third_best_index[row] = second_best_index[row]
            second_best_val[row] = tmp
            second_best_index[row] = i
        elif(tmp>third_best_val[row]):
            third_best_val[row] = tmp
            third_best_index[row] = i
        print(i)
        
        
        
# Device Details
        
first_val = [0.00000000002]*n
first_val = np.asarray(first_val)
second_val = [0.00000000002]*n
second_val = np.asarray(second_val)
third_val = [0.00000000002]*n
third_val = np.asarray(third_val)
first_index = [1]*n
first_index = np.asarray(first_index)
second_index = [1]*n
second_index = np.asarray(second_index)
third_index = [1]*n
third_index = np.asarray(third_index)       

d_x = cuda.to_device(X)
d_first_val = cuda.device_array_like(first_val)
d_first_index = cuda.device_array_like(first_index)
d_second_val = cuda.device_array_like(second_val)
d_second_index = cuda.device_array_like(second_index)
d_third_val = cuda.device_array_like(third_val)
d_third_index = cuda.device_array_like(third_index)
    
device = cuda.get_current_device()
tpb = device.WARP_SIZE       #blocksize or thread per block
bpg = int(np.ceil((n)/tpb))  #block per grid

cuda_dist[bpg,tpb](d_x,d_first_val,d_first_index,d_second_val,d_second_index,d_third_val,d_third_index) #calling the kernel

# Transfer output from device to host
first_val = d_first_val.copy_to_host()
print (first_val[:10]) # First 10 values
# Transfer output from device to host
first_index = d_first_index.copy_to_host()
print (first_index[:10]) # First 10 indexes