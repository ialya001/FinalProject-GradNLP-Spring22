#!/usr/bin/env python
# coding: utf-8

# In[27]:


#Import necessarily libraries
import numpy as np
import pandas as pd
import re
import time
from datasketch import MinHash, MinHashLSHForest


# In[28]:


#Preprocess will split a string of text into individual tokens/shingles based on whitespace.
def preprocess(text):
    text = re.sub(r'[^\w\s]','',text)
    tokens = text.lower()
    tokens = tokens.split()
    return tokens


# # First Layer Locallity Sensitive Hashing
# https://www.learndatasci.com/tutorials/building-recommendation-engine-locality-sensitive-hashing-lsh-python/

# In[29]:


#Number of Permutations
permutations = 100


# In[30]:


def get_forest(data, perms):
    start_time = time.time()
    
    minhash = []
    
    for text in data['text']:
        tokens = preprocess(text)
        m = MinHash(num_perm=perms)
        for s in tokens:
            m.update(s.encode('utf8'))
        minhash.append(m)
        
    forest = MinHashLSHForest(num_perm=perms)
    
    for i,m in enumerate(minhash):
        forest.add(i,m)
        
    forest.index()
    
    print('It took %s seconds to build forest.' %(time.time()-start_time))
    
    return forest


# In[31]:


def predict(text, database, perms, num_results, forest):
    start_time = time.time()
    
    tokens = preprocess(text)
    m = MinHash(num_perm=perms)
    for s in tokens:
        m.update(s.encode('utf8'))
        
    idx_array = np.array(forest.query(m, num_results))
    if len(idx_array) == 0:
        return None # if your query is empty, return none
    
    result = database.iloc[idx_array]['question2']
    
    print('It took %s seconds to query forest.' %(time.time()-start_time))
    
    return result


# # Upload the Dataset

# In[32]:


db = pd.read_csv('/Users/ibrahim/Desktop/CAP5640/FinalProject/Dataset/train.csv')#Please change this to your folder location
db['text']= db['question2']
#print(db)


# In[33]:


db=db.head(404)#Read the number of rows you would like from the train dataset
#print(db)404289


# In[34]:


forest = get_forest(db, permutations)


# In[35]:


num_recommendations = 100 #We would like to get the best 100 candidates
query = "How do you get deleted Instagram chats?"
result = predict(query, db, permutations, num_recommendations, forest)
print('\n Top Recommendation(s) is(are) \n', result)


# In[36]:


#Create a list of candiates to be taken to the next layer which is Cosine Simialrity
candidates=[]
candidates=result.values


# In[37]:


print(candidates)


# # Second Layer BERT Vectors+Cosine Similarity
# https://www.analyticsvidhya.com/blog/2020/08/top-4-sentence-embedding-techniques-using-python/

# In[38]:


#Tokenize the candidates to be used in Bert Vector model
import nltk
from nltk.tokenize import word_tokenize
tokenized_sent = []
for s in candidates:
    tokenized_sent.append(word_tokenize(s.lower()))


# In[39]:


import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np


# In[40]:


def cosine(u, v):#Cosine Similarity Calculation
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


# In[41]:


from sentence_transformers import SentenceTransformer #Vectorize the sentences using bert
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')


# In[42]:


sentence_embeddings = sbert_model.encode(candidates)

#print('Sample BERT embedding vector - length', len(sentence_embeddings[0]))
#print('Sample BERT embedding vector - note includes negative values', sentence_embeddings[0])


# In[43]:


query_vec = sbert_model.encode([query])[0]#Vectorize the query using bert 


# In[44]:


Cosine_Candidates={} #This dicitonary to get the candidates and thier cosine similarity
for sent in candidates:
    sim = cosine(query_vec, sbert_model.encode([sent])[0])
    #print("Sentence = ", sent, "; similarity = ", sim)
    Cosine_Candidates[sent]=[sim]


# In[45]:


import operator#To sort the candidates from the highest to the lowest
sorted_d = dict(sorted(Cosine_Candidates.items(), key=operator.itemgetter(1),reverse=True))
print(sorted_d)


# In[46]:


NW_Candidates2=[] #The questions that has 85% similarity only
for i in sorted_d:
    if sorted_d[i] >= [0.85]:
        NW_Candidates2.append(i)


# In[47]:


#sort the dictionary in descending way and get the best 10 possible similar questions
NW_Candidates=NW_Candidates2[:10]


# In[48]:


print(NW_Candidates)
#print(len(NW_Candidates))


# In[49]:


if not NW_Candidates: #To check if there is no similar question detected after Cosine layer
        print("Since we did not get any candidates(i.e list of possible questions) from our Cosine Similarity, therefore, the question is not duplicated ")


# # Third Layer - Needleman-Wunch Algorith 
# https://github.com/scastlara/minineedle

# In[50]:


from minineedle import needle, core

R = {} # Dicitonary to Save query, target sentence, precent identity and the actual aligmnet
for i in NW_Candidates:
    print("--------------",i)
    i.lower()
    query.lower()
    i.split()
    query.split()
    alignment = needle.NeedlemanWunsch(query,i)
    x = alignment.get_identity()
    y = alignment 
    R[i]=[x , query, i, y]


# In[51]:


if R:
    max_value = max(R.values())
    print('Query: ',query,'\n')
    print('Target Question: \n',max_value[2],'\n')
    print('Perecent Identity using Needleman-Wunch Algorithm:',max_value[0])
    print(max_value[3])
else:
    print("Your Needleman-Wunch is empty")


# # Finialize the Results and Possible Candidates if found

# In[53]:


print('The question a user asked is:\n')
print("-",query,"\n")
#If Cosine candidates is empty then this question is not duplicated
#print(len(NW_Candidates))
if not NW_Candidates:
        print("Congratulations 🎉 🎊 🍾 🎈 your question has never been asked before")        
else:
    print("*Your question was asked before, this is the best candidate(s) question we found:\n")
    for i in NW_Candidates:
        print("- ",i)
    print("\nAccording to our model we belive that the best candidate is:\n")
    print("→",max_value[2]) 


# In[ ]:




