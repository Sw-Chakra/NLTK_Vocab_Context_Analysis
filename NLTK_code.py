
# coding: utf-8

# In[1]:


import nltk
t = nltk.download()


# In[2]:


import os, sys
import nltk
import itertools
import string
import warnings
import numpy as np
from nltk.corpus import brown, stopwords
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# In[3]:


class WordEmbeddingsUtil(object):

    @staticmethod
    def get_brown_words():
       raw_brown_words = [str(word.lower()) for word in brown.words()] # Converts to lower string
       stop_words = [str(word.lower()) for word in stopwords.words('english')]
       stop_words.append(string.punctuation)
       return [word for word in raw_brown_words if word not in stop_words and word.isalnum()]

    @staticmethod
    def get_common_vocab_words(word_list, common_count, vocab_count):
       freq_dist = nltk.FreqDist(word_list)
       vocab_count_dict = {}
       vocab_words = []
       common_tuple = freq_dist.most_common(common_count)
       common_words = [tup[0] for tup in common_tuple]
       vocab_tuple = freq_dist.most_common(vocab_count)
       for tup in vocab_tuple:
           vocab_words.append(tup[0])
           vocab_count_dict[tup[0]] = tup[1]
       return common_words, vocab_words, vocab_count_dict

    @staticmethod
    def get_four_grams( word_list, vocab_words):
       four_grams = []
       all_tuples = zip(word_list, word_list[1:],word_list[2:],word_list[3:],word_list[4:])
       for tup in all_tuples:
           if tup[2] in vocab_words:
               four_grams.append(tup)
       return four_grams

    @staticmethod
    def create_vocab_common_dict( vocab, common):
       vocab_common_keys = list(itertools.product(
           vocab, common))
       return dict.fromkeys(vocab_common_keys, 0.0)

    @staticmethod
    def find_vocab_occurrence(vocab_dict, four_gram_tuple, common):
       for four_gram in four_gram_tuple:
           for word in four_gram:
               if word in common:
                   vocab_dict[(four_gram[2], word)] += 1.0
                   print ("Found : " + str((four_gram[2], word)))
       return vocab_dict

    @staticmethod
    def store_dict_file(vocab_dict, filename):
       np.save(filename, vocab_dict)

    @staticmethod
    def load_dict_file(filename):
       return np.load(filename).item()



# In[4]:


class WordCluster(object):

   @staticmethod
   def pca_dim_reduce(phi_vector, reduce_dim):
       pca = PCA(n_components=reduce_dim).fit(phi_vector)
       pca = PCA(n_components=reduce_dim).fit_transform(phi_vector)
       return pca

   @staticmethod
   def kmeans_clustering(vector, cluster_size):
       kmeans = KMeans(n_clusters=cluster_size).fit(vector)
       return kmeans

   @staticmethod
   def cosine_dist(vector1, vector2):
       vector1 = np.asarray(vector1)
       vector2 = np.asarray(vector2)
       distance = 1.0 - (np.dot(vector1.T, vector2)) / (np.linalg.norm(
           vector1) * np.linalg.norm(vector2))
       return distance


# In[5]:


common_count = 1000
vocab_count = 5000
reduced_dim = 100

cluster_size = 100
dict_file = 'vocab_dict.npy'
reuse_prob_dict = True
we = WordEmbeddingsUtil()
wc = WordCluster()



# Get Brown corpus words
print ("Get Brown corpus words")
brown_words = we.get_brown_words() # storing list of brown words - all words except stop words(1.16M- stop words= 515882 words)


 
# Get common words and vocabulary words
print ("Get common words and vocabulary words")
brown_common, brown_vocab, vocab_count_dict = we.get_common_vocab_words(brown_words, common_count,vocab_count)


# In[ ]:


# Build Four-grams from the corpus list
print ("Build Four-grams from the corpus list")
brown_four_gram = we.get_four_grams(brown_words, brown_vocab) # returning a set of w1,w2,w,w3,w4...where w is one of the 5000 words

#  Create a dictionary from vocabulary and common
print ("Create a dictionary from vocabulary and common")
vocab_dict = we.create_vocab_common_dict(brown_vocab, brown_common) # 5000*1000 vc pairs (in each row two elements)

# Populate dictionary with occurrence count
print ("Populate dictionary with occurrence count")
vocab_dict = we.find_vocab_occurrence(vocab_dict, brown_four_gram, brown_common)

# Store dictionary in a file for future use
print ("Storing dictionary in NPY file")
we.store_dict_file(vocab_dict, dict_file)


# In[6]:


# Load dictionary from preprocessed file
# Returns dictionary with { (vocab, common) : [count] }
print ("Loading dictionary from NPY file")
vocab_dict = we.load_dict_file(dict_file)


# In[7]:


# Compute P(c|w) = N(c,w) / N(w)
print ("Computing Probability of C given W")
prob_common_given_vocab = {}
for key in vocab_dict.keys():
    prob_common_given_vocab[key] = float(vocab_dict[key]) / float(vocab_count_dict[key[0]])


# In[8]:


# Compute P(c) = N(c) / N(total words)
print ("Computing Probability of C")
prob_common = {}
for word in brown_common:
    prob_common[word] = float(vocab_count_dict[word]) / float(len(brown_words))


# In[9]:


# PHI vector for each word
print ("Building PHI vector for each vocab word")
phi_vocab = {}
for word in brown_vocab:
    phi_vocab[word] = []
    for common in brown_common:
        if prob_common_given_vocab[(word, common)] == 0:
            log_cw_c = 0
        else:
            log_cw_c = np.log(prob_common_given_vocab[(
                word,common)]/prob_common[common])
        phi_vocab[word].append(max(0,log_cw_c))
        phi_vocab[word].append(max(0,log_cw_c))

 


# In[10]:


# Converting dictionary to list for dimensionality reduction
print ("Converting dictionary to list")
word_list = []
vector_list = []
#TODO: Figure out how to concatenate arrays vertically !!
for word, vector in phi_vocab.items():
    word_list.append(word)
    vector_list.append(vector)


# In[11]:


# Applying PCA for dimensionality reduction
print ("Applying PCA")
reduced_vector = wc.pca_dim_reduce(vector_list, reduced_dim)


# In[15]:


# Cluster the reduced vector using K-Means Clustering
print ("Applying K Means Clustering")
kmeans = wc.kmeans_clustering(reduced_vector, cluster_size)


# In[16]:


# Grouping vocabulary based on clusters
print ("Grouping vocabulary based on labels")
word_clusters = {}
for idx in range(len(kmeans.labels_)):
    if kmeans.labels_[idx] not in word_clusters.keys():
        word_clusters[kmeans.labels_[idx]] = [word_list[idx]]
    else:


        #word_list = np.append(word_list, word, axis = 0)
        #vector_list = np.append(vector_list, vector, axis = 0)
        word_clusters[kmeans.labels_[idx]].append(word_list[idx])
#print(word_clusters)
test_words = ['communism', 'autumn', 'cigarette', 'pulmonary',
              'mankind', 'africa', 'chicago', 'revolution', 'september',
              'chemical', 'detergent', 'dictionary', 'storm', 'worship',
              'husband','hospital','international','angel',
             'slowly','nuclear','justice','building','brown','administration','equipment']


# In[17]:


# Finding Nearest neighbour for list of test words
print ("Finding nearest neighbour for test words")
for test_word in test_words:
    test_label = kmeans.predict(np.array(reduced_vector[word_list.index(test_word)]).reshape(1,-1))
    min_dist = float("inf")
    close_word = 'NONE'
    cluster_words = word_clusters[test_label[0]]
    for cluster_word in cluster_words:
        dist = wc.cosine_dist(reduced_vector[word_list.index(test_word)],
                              reduced_vector[word_list.index(cluster_word)])
        if (dist < min_dist) and (cluster_word != test_word):
            min_dist = dist
            close_word = cluster_word
    print (str(test_word) + " is near to " + str(close_word))

print (len(word_list))
print (len(vector_list))
print ("Completed..")


