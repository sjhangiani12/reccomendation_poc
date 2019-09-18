
# # Document Retrieval using TF-IDF Weighted Rank and TF-IDF Cosine Similarity

# ## Imports

import math
import re
import pickle
import pandas as pd
import copy
import numpy as np
import string
import os
from num2words import num2words
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')


print('loading and parsing data')
alpha = 0.3
folders = [x[0] for x in os.walk(str(os.getcwd() + '/data/stories/'))]
folders[0] = folders[0][:len(folders[0])-1]

# ## Collecting the file names and titles
dataset = []

c = False
for i in folders:
    file = open(i+"/index.html", 'r')
    text = file.read().strip()
    file.close()

    file_name = re.findall('><A HREF="(.*)">', text)
    file_title = re.findall('<BR><TD> (.*)\n', text)

    if c == False:
        file_name = file_name[2:]
        c = True
    for j in range(len(file_name)):
        dataset.append((str(i) + "/" + str(file_name[j]), file_title[j]))

print('data loaded.')


def parse_doc(id):
    file = open(dataset[id][0], 'r', encoding='cp1250')
    text = file.read().strip()
    file.close()
    return(text)


# # Preprocessing

def convert_lower_case(data):
    return np.char.lower(data)


def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text


def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data


def remove_apostrophe(data):
    return np.char.replace(data, "'", "")


def stemming(data):
    stemmer = PorterStemmer()

    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text


def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text


def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data)  # remove comma seperately
    data = remove_apostrophe(data)
    data = remove_stop_words(data)
    data = convert_numbers(data)
    data = stemming(data)
    data = remove_punctuation(data)
    data = convert_numbers(data)
    data = stemming(data)  # needed again as we need to stem the words
    # needed again as num2word is giving few hypens and commas fourty-one
    data = remove_punctuation(data)
    # needed again as num2word is giving stop words 101 - one hundred and one
    data = remove_stop_words(data)
    return data


print('preprocessing/prepping data.')
# ## Extracting Data
processed_text = []
processed_title = []
texts = []
titles = []

for i in dataset[:len(dataset)]:
    file = open(i[0], 'r', encoding="utf8", errors='ignore')
    text = file.read().strip()
    file.close()

    titles.append(i[1])
    texts.append(text)
    processed_text.append(word_tokenize(str(preprocess(text))))
    processed_title.append(word_tokenize(str(preprocess(i[1]))))

df = pd.DataFrame()
df['titles'] = titles
df['text'] = texts
df['processed_text'] = processed_text
df['processed_title'] = processed_title
df.head()

print('data prepped.')

print('applying tf-idf conversion')
# ## Calculating DF for all words

doc_freq = {}

for i in range(len(df)):
    tokens = df.processed_text[i]
    for w in tokens:
        try:
            doc_freq[w].add(i)
        except:
            doc_freq[w] = {i}

    tokens = df.processed_title[i]
    for w in tokens:
        try:
            doc_freq[w].add(i)
        except:
            doc_freq[w] = {i}
for i in doc_freq:
    doc_freq[i] = len(doc_freq[i])


total_vocab = [x for x in doc_freq]


def get_doc_freq(word):
    try:
        c = doc_freq[word]
        return c
    except:
        return 0


# ### Calculating TF-IDF for body, we will consider this as the actual tf-idf as we will add the title weight to this.

tf_idf = {}
for i in range(len(df)):

    tokens = df.processed_text[i]
    counter = Counter(tokens + df.processed_title[i])
    words_count = len(tokens + df.processed_title[i])

    for token in np.unique(tokens):

        tf = counter[token]/words_count
        document_freq = get_doc_freq(token)
        idf = np.log((len(df)+1)/(document_freq+1))

        tf_idf[df.titles[i], token] = tf*idf


# calulcating tf_idf for the title
tf_idf_title = {}
for i in range(len(df)):

    tokens = processed_title[i]
    counter = Counter(tokens + processed_text[i])
    words_count = len(tokens + processed_text[i])

    for token in np.unique(tokens):

        tf = counter[token]/words_count
        document_freq = get_doc_freq(token)
        # numerator is added 1 to avoid negative values
        idf = np.log((len(df)+1)/(document_freq+1))

        tf_idf_title[df.titles[i], token] = tf*idf


# ## Merging the TF-IDF according to weights
for i in tf_idf:
    tf_idf[i] *= alpha
for i in tf_idf_title:
    tf_idf[i] = tf_idf_title[i]
print('calculated tf-idf.')
print('Reccomendation, assemble. Its go time')


# # TF-IDF Matching Score Ranking
def matching_score(num_responses, query):
    assert type(num_responses) == int
    if (num_responses > 11) or (num_responses < 0):
        raise ValueError('Privacy dial setting ranges from 0 to 11')
    assert type(query) == str

    preprocessed_query = preprocess(query)
    tokens = word_tokenize(str(preprocessed_query))
    query_weights = {}

    for key in tf_idf:

        if key[1] in tokens:
            try:
                query_weights[key[0]] += tf_idf[key]
            except:
                query_weights[key[0]] = tf_idf[key]

    query_weights = sorted(query_weights.items(),
                           key=lambda x: x[1], reverse=True)
    list_of_titles = []

    for i in query_weights[:num_responses]:
        print(i[0])
        list_of_titles.append(i[0])

    books = []
    # for book in list_of_titles:

    return query, list_of_titles


matching_score(3, "love")
