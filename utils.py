#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 16:21:05 2022

@author: lacopoginhassi
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer

import string
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.corpus import stopwords as stop_words
from nltk.corpus import wordnet

import pprint
import re

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def class_tfidf(topic_clusters, vocab):
  topic_matrix = np.zeros((len(topic_clusters), len(vocab)))

  for t in topic_clusters:
    for w in topic_clusters[t]:
      topic_matrix[t,w] = topic_clusters[t][w]

  A = np.mean(np.sum(topic_matrix>0, axis=1))

  topic_matrix_norm = topic_matrix*np.log(1+(A/topic_matrix.sum(axis=0)))

  return topic_matrix_norm, topic_matrix


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def top_k_topic(topic, topic_matrix, vocab, k=10, pt = True):
  soft = nn.Softmax(0)
  if pt:
    topk_values, topk_idxs = torch.topk(soft(topic_matrix[topic]), k)
    topk_values = topk_values.detach().cpu().tolist()
    topk_idxs = topk_idxs.detach().cpu().tolist()
  else:
    probs = softmax(topic_matrix[topic])
    topk_idxs = np.argpartition(probs, -k)[-k:]
    topk_values = probs[topk_idxs]
  top_words = {}
  for value, idx in zip(topk_values, topk_idxs):
    top_words[vocab[idx]] = value
  
  return top_words


def print_all_topics(topic_matrix, vocab, k=10, pt = True):
  for i in range(topic_matrix.shape[0]):
    print(f'Top {k} words for Topic n. {i}')
    pprint.pprint(top_k_topic(i, topic_matrix, vocab, k, pt))
    print('=======================\n')

def readGloveFile(gloveFile):
    with open(gloveFile, 'r') as f:
        wordToGlove = {}  
        wordToIndex = {}  
        indexToWord = {}  

        for line in f:
            record = line.strip().split()
            token = record[0] 
            wordToGlove[token] = np.array(record[1:], dtype=np.float64) 
            
        tokens = sorted(wordToGlove.keys())
        for idx, tok in enumerate(tokens):
            kerasIdx = idx + 1  
            wordToIndex[tok] = kerasIdx 
            indexToWord[kerasIdx] = tok 

    return wordToIndex, indexToWord, wordToGlove


def w2iGLOVE(text, wordToIndex, no_token = False):
  if no_token:
    tokens = text
  else:
    tokens = nltk.word_tokenize(text)
  idxs = []
  for t in tokens:
    if t not in wordToIndex:
      # idxs.append(wordToIndex["#"])
      idxs.append(2)
    else:
      idxs.append(wordToIndex[t])
  return idxs


def get_wordnet_pos(tag):
    """Map POS tag to first character lemmatize() accepts"""
    
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)
    
def lemmatize(doc, return_token = False):
    pos_tagged_words = nltk.pos_tag(nltk.word_tokenize(doc))
    
    return ' '.join([lemmatizer.lemmatize(x[0], get_wordnet_pos(x[1])) for x in pos_tagged_words])
    
class WhiteSpacePreprocessing():
    """
    Provides a very simple preprocessing script that filters infrequent tokens from text.
    Code from https://contextualized-topic-models.readthedocs.io/en/latest/index.html
    """
    def __init__(self, documents, stopwords_language="english", vocabulary_size=2000):
        """
        :param documents: list of strings
        :param stopwords_language: string of the language of the stopwords (see nltk stopwords)
        :param vocabulary_size: the number of most frequent words to include in the documents. Infrequent words will be discarded from the list of preprocessed documents
        """
        self.documents = documents
        self.stopwords = set(stop_words.words(stopwords_language))
        self.vocabulary_size = vocabulary_size

    def preprocess(self, just_alpha = True, min_length = 0, lemma = False):
        """
        Note that if after filtering some documents do not contain words we remove them. That is why we return also the
        list of unpreprocessed documents.
        :return: preprocessed documents, unpreprocessed documents and the vocabulary list
        """
        preprocessed_docs_tmp = self.documents
        preprocessed_docs_tmp = [doc.lower() for doc in preprocessed_docs_tmp]
        if just_alpha:
            preprocessed_docs_tmp = [re.sub('[^a-z\s]', '', doc) for doc in preprocessed_docs_tmp]
        else:
            preprocessed_docs_tmp = [doc.translate(
                str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in preprocessed_docs_tmp]
        preprocessed_docs_tmp = [' '.join([w for w in doc.split() if len(w) > min_length and w not in self.stopwords])
                             for doc in preprocessed_docs_tmp]
        
        if lemma:
            preprocessed_docs_tmp = [lemmatize(x) for x in preprocessed_docs_tmp]
        
        vectorizer = CountVectorizer(max_features=self.vocabulary_size, token_pattern=r'\b[a-zA-Z]{2,}\b')
        vectorizer.fit_transform(preprocessed_docs_tmp)
        vocabulary = set(vectorizer.get_feature_names_out())
        preprocessed_docs_tmp = [' '.join([w for w in doc.split() if w in vocabulary])
                                 for doc in preprocessed_docs_tmp]

        preprocessed_docs, unpreprocessed_docs = [], []
        for i, doc in enumerate(preprocessed_docs_tmp):
            if len(doc) > 0:
                preprocessed_docs.append(doc)
                unpreprocessed_docs.append(self.documents[i])

        return preprocessed_docs, unpreprocessed_docs, list(vocabulary)
        
        
def preprocess_default(data, just_alpha = True, min_length = 0, lemma = False):
    
    sp = WhiteSpacePreprocessing(data)
          
    data_preprocessed, data, vocab = sp.preprocess(just_alpha, min_length, lemma)
    
    return data_preprocessed, data