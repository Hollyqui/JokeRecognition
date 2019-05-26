import json
import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
import itertools
from collections import Counter

stop_words=set(nltk.corpus.stopwords.words('english'))
stop_words = stopwords.words('english')

tokenizer = RegexpTokenizer(r'\w+')
porter=PorterStemmer()
def stem_sentence(sentence):
    token = tokenizer.tokenize(sentence)
    token = [word for word in token if word not in stop_words]
    stemmed = []
    for word in token:
        stemmed.append(word)
        stemmed.append(" ")
    return "".join(stemmed)

data = pd.read_csv('abcnews-date-text.csv')
data = data['headline_text']
jokes = []
for i in range(800):
    jokes.append(data[i])


for i in range(len(jokes)):
    jokes[i] = stem_sentence(jokes[i])
## word to vec
embeddings_index = {}
f = open('glove.6B.200d.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

vectorized = []
sequence_length = 149
temp = 0
for sentence in jokes:
    vectorized.append([])
    for word in sentence:
        vectorized[-1].append(np.array(embeddings_index.get(word)).tolist())
        if vectorized[-1][-1] is None or len(vectorized[-1][-1]) < 200:
            vectorized[-1][-1] = np.zeros(200).tolist()
            temp += 1
            print(temp)
    for i in range(sequence_length - len(sentence)):
        vectorized[-1].append(np.zeros(200).tolist())
np.save('headlines_vec', np.array(vectorized))
