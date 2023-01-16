import os
# if the following command generates an error, you probably didn't enable 
# the cluster security option "Allow API access to all Google Cloud services"
# under Manage Security → Project Access when setting up the cluster
os.system('gcloud dataproc clusters list --region us-central1')
os.system('pip install -q google-cloud-storage==1.43.0')
os.system('pip install flask')
os.system(' pip3 install --user lemminflec')

from InvertedIndex import InvertedIndex
from Processor import Processor
from WikiFileLoader import WikiFileLoader


import numpy as np
import pandas as pd
from functools import partial
from collections import Counter, OrderedDict, defaultdict
import pickle
import heapq
from itertools import islice, count, groupby
import itertools
from xml.etree import ElementTree
import codecs
import csv
import re
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from pathlib import Path
import hashlib
import bz2
import time
from pathlib import Path
import string 
import random
import json
from contextlib import closing
import uuid
nltk.download('punkt')
nltk.download("stopwords")
from flask import Flask, request, jsonify
nltk.download("wordnet")
nltk.download("omw-1.4")
from nltk.stem import WordNetLemmatizer
from google.cloud import storage
from lemminflect import getLemma
def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()


from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf, SparkFiles
from pyspark.sql import SQLContext
from graphframes import *
spark = SparkSession.builder.master("local[*]").appName("PySparkShell").getOrCreate()
from pyspark import SparkContext


import json
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links", 
                    "may", "first", "see", "history", "people", "one", "two", 
                    "part", "thumb", "including", "second", "following", 
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
# RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

NUM_BUCKETS = 124
def token2bucket_id(token):
    return int(_hash(token),16) % NUM_BUCKETS

def parse_parquet_file(parquetFile):
    print("Preproccessor -> parse_parquet_file : Started")
    """
    Parsing the perquet file into lists of pairs
    ----------
    Parameters
    ----------
    parquet_file: parquet file of all curpos files
    ----------
    Returns
    doc_text_pairs: rdd of pairs
    <key -> doc_id, value -> text>
    doc_title_pairs: rdd of pairs
    <key -> doc_id, value -> title>
    doc_anchor_pairs: rdd of pairs
    <key -> doc_id, value -> anchor_text>
    """
    doc_text_pairs = parquetFile.select("id", "text").rdd
    doc_title_pairs = parquetFile.select("id", "title").rdd
    doc_anchor_pairs = parquetFile.select("id", "anchor_text").rdd

    return doc_text_pairs, doc_title_pairs, doc_anchor_pairs


def parse_parquet_file_text(parquetFile):
    print("Preproccessor -> parse_parquet_file : Started")
    """
    Parsing the perquet file into lists of pairs
    ----------
    Parameters
    ----------
    parquet_file: parquet file of all curpos files
    ----------
    Returns
    doc_text_pairs: rdd of pairs
    <key -> doc_id, value -> text>
    doc_title_pairs: rdd of pairs
    <key -> doc_id, value -> title>
    doc_anchor_pairs: rdd of pairs
    <key -> doc_id, value -> anchor_text>
    """
    doc_text_pairs = parquetFile.select("id", "text").rdd

    return doc_text_pairs


def parse_parquet_file_title(parquetFile):
    print("Preproccessor -> parse_parquet_file : Started")
    """
    Parsing the perquet file into lists of pairs
    ----------
    Parameters
    ----------
    parquet_file: parquet file of all curpos files
    ----------
    Returns
    doc_text_pairs: rdd of pairs
    <key -> doc_id, value -> text>
    doc_title_pairs: rdd of pairs
    <key -> doc_id, value -> title>
    doc_anchor_pairs: rdd of pairs
    <key -> doc_id, value -> anchor_text>
    """
    doc_title_pairs = parquetFile.select("id", "title").rdd

    return doc_title_pairs

def parse_parquet_file_anchor(parquetFile):
    print("Preproccessor -> parse_parquet_file : Started")
    """
    Parsing the perquet file into lists of pairs
    ----------
    Parameters
    ----------
    parquet_file: parquet file of all curpos files
    ----------
    Returns
    doc_text_pairs: rdd of pairs
    <key -> doc_id, value -> text>
    doc_title_pairs: rdd of pairs
    <key -> doc_id, value -> title>
    doc_anchor_pairs: rdd of pairs
    <key -> doc_id, value -> anchor_text>
    """
    return parquetFile.select("id", "anchor_text").rdd




def anchor_to_disk(anchors_rdd,dir):
    anchor_id = uuid.uuid4()
    file_path = f"{dir}{anchor_id}"
    if os.path.exists(file_path):
        os.remove(file_path) 
    anchors_rdd.saveAsPickleFile(file_path)


def word_count(id, text):
    
    ''' Count the frequency of each word in `text` (tf) that is not included in 
    `all_stopwords` and return entries that will go into our posting lists. 
    Parameters:
    -----------
    text: str
      Text of one document
    id: int
      Document id
    Returns:
    --------
    List of tuples
      A list of (token, (doc_id, tf)) pairs 
      for example: [("Anarchism", (12, 5)), ...]
    '''
    preTokens = Processor.tokenize(text)
    tokens = lemmatize(preTokens)
    freqTuples = []
    tokenCount = {}
    for token in tokens:
        if (token not in all_stopwords and token not in string.punctuation):
            if token in tokenCount.keys():
                tokenCount[token] += 1
            else:
                tokenCount[token] = 1
    for key, val in tokenCount.items():
        freqTuples.append((key, (id, val)))
    return freqTuples


def reduce_word_counts(unsorted_pl):
    ''' Returns a sorted posting list by wiki_id.
    Parameters:
    -----------
    unsorted_pl: list of tuples
      A list of (wiki_id, tf) tuples 
    Returns:
    --------
    list of tuples
      A sorted posting list.
    '''
    sortedWords = sorted(unsorted_pl)
    return sortedWords
    
def calculate_df(postings):
    ''' Takes a posting list RDD and calculate the df for each token.
    Parameters:
    -----------
    postings: RDD
      An RDD where each element is a (token, posting_list) pair.
    Returns:
    --------
    RDD
      An RDD where each element is a (token, df) pair.
    '''
    return postings.mapValues(lambda x : len(x))

def partition_postings_and_write(postings):
    ''' A function that partitions the posting lists into buckets, writes out 
    all posting lists in a bucket to disk, and returns the posting locations for 
    each bucket. Partitioning should be done through the use of `token2bucket` 
    above. Writing to disk should use the function  `write_a_posting_list`, a 
    static method implemented in inverted_index_colab.py under the InvertedIndex 
    class. 
    Parameters:
    -----------
    postings: RDD
      An RDD where each item is a (w, posting_list) pair.
    Returns:
    --------
    RDD
      An RDD where each item is a posting locations dictionary for a bucket. The
      posting locations maintain a list for each word of file locations and 
      offsets its posting list was written to. See `write_a_posting_list` for 
      more details.
    '''
    b = postings.map(lambda w: (token2bucket_id(w[0]),[w]))
    b = b.reduceByKey(lambda x,y: x+y)
    return b.map(lambda x: InvertedIndex.write_a_posting_list(x,"shemiperetz-irpro"))
    
    
# def tfidf_vec_size(id, text,N,w2df, stopwords):
#     tokens = [token.group() for token in RE_WORD.finditer(str(text).lower())]
#     w2tf =  Counter(tokens)
#     sqeure_tfidf_sum = 0
#     epsilon = 10 ** -9
#     for token, tf in w2tf.items():
#         if token not in stopwords:
#             try:
#                 df = w2df[token] + epsilon
#             except:
#                 df = 0 + epsilon
                
#             idf = np.log(N / df)
#             tfidf = tf  * np.log(N / idf)
#             sqeure_tfidf_sum += tfidf ** 2
#     tfidf_vec_size = np.sqrt(sqeure_tfidf_sum)
    
#     return tfidf_vec_size


# def read_index_tfidf_sizes(base_dir, name):
#     return sc.pickleFile(base_dir, name)

# def reduce_anchors(word, lst):
#     d ={}
    
#     tokens = [token.group() for token in RE_WORD.finditer(str(word).lower())]
#     if len(tokens) > 0:
#         if (tokens[0] not in all_stopwords):
#             token = tokens[0]
#         else:
#             return None
#     else:
#         return None
    
#     final_d = {token: None}
#     if isinstance(lst,int) or isinstance(lst,int):
#         return None
#     for tup in lst:
#         d.setdefault(tup[0],0)
#         d[tup[0]] += 1
#     final_d[token] = d
#     return final_d

def lemmatize(words):
    lem_word = set([getLemma(f'{word}', upos='VERB') for word in words])
    return lem_word



os.system('mkdir inv_index_title_lem inv_index_text_lem')

wiki_loader = WikiFileLoader()
parquetFile = wiki_loader.get_all_files_from_bucket()

panc = set(string.punctuation)
stopwords_set = stopwords.words('english')
doc_text_pairs = parse_parquet_file_text(parquetFile)

text_word_counts = doc_text_pairs.flatMap(lambda x: word_count(x[0], x[1]))


postings = text_word_counts.groupByKey().mapValues(reduce_word_counts)
# filtering postings and calculate df
postings_filtered = postings.filter(lambda x: len(x[1])>50)
w2df = calculate_df(postings_filtered)
w2df_dict = w2df.collectAsMap()
N = doc_text_pairs.count()


# partition posting lists and write out
_ = partition_postings_and_write(postings_filtered).collect()


# collect all posting lists locations into one super-set
from google.cloud import storage
bucket_name = 'shemiperetz-irproject' 
super_posting_locs = defaultdict(list)
client = storage.Client()
for blob in client.list_blobs(bucket_name, prefix='postings_gcp'):
    if not blob.name.endswith("pickle"):
        continue
    with blob.open("rb") as f:
        posting_locs = pickle.load(f)
        for k, v in posting_locs.items():
            super_posting_locs[k].extend(v)


# Create inverted index instance
inverted = InvertedIndex()
# Adding the posting locations dictionary to the inverted index
inverted.posting_locs = super_posting_locs
# Add the token - df dictionary to the inverted index
inverted.df = w2df_dict
inverted.N = N
# inverted.doc2tfidf_size = doc2tfidf_size  TODO : SAVE elsewhere
# write the global stats out
inverted.write_globals('.', 'inv_index_text_lem')
# upload to gs
index_src = "inv_index_text_lem.pkl"
index_dst = f'gs://{bucket_name}/postings_gcp/{index_src}'
import os 
os.system(f"gsutil cp ${index_src} ${index_dst}")

