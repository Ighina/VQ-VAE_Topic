#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 16:25:15 2022

@author: lacopoginhassi
"""
import os
from utils import *
import torch

from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer

class WordsDataset(Dataset):
  def __init__(self, sentences, targets, wordToIndex, no_token=False):
    self.sentences = []
    self.targets = []
    
    for index, sentence in enumerate(sentences):
        if len(sentence)>0 and len(targets[index])>0:
            self.sentences.append(w2iGLOVE(sentence, wordToIndex, no_token))
            self.targets.append(targets[index].tolist())
    
    if len(self.targets)!=len(self.sentences):
        raise ValueError("Sentences and Targets have different lengths!")

  def __getitem__(self, index):
        return {
            'id': torch.tensor(index),
            'source': torch.tensor(self.sentences[index], dtype = torch.long),
            'target': torch.tensor(self.targets[index], dtype = torch.float)
        }
        
  def __len__(self):
        return len(self.sentences)

  @staticmethod
  def merge(batch):
    lengths = [len(b) for b in batch]
    max_length = max(lengths)

    padded_batch = torch.zeros((len(batch), max_length), dtype = torch.long)
    for index, length in enumerate(lengths):
      padded_batch[index,:length] = batch[index]

    return padded_batch

  def collater(self, samples):
    source = self.merge([s['source'] for s in samples])
    target = torch.stack([s['target'] for s in samples])
    lengths = torch.LongTensor([len(s['source']) for s in samples])

    return source, target, lengths

def load_data(use_glove, preprocess, 
              data_directory = 'data',
              dataset = 'ng20',
              encoder = None,
              just_alpha = False,
              min_length = 0,
              max_target = 2000,
              lemmatize = False,
              batch_size = 50,
              glove_file = 'glove.6B.300d.txt'):
    
    if dataset=='ng20':
        if preprocess:
            no_token = False
            from sklearn.datasets import fetch_20newsgroups
            
            train_data = fetch_20newsgroups(subset='train')['data']
            test_data = fetch_20newsgroups(subset='test')['data']
          
            valid_data = test_data[-len(test_data)//20:]
            test_data = test_data[:len(test_data)-len(test_data)//20]
          
            train_data_preprocessed, train_data = preprocess_default(train_data, just_alpha, min_length, lemmatize)
          
            valid_data_preprocessed, valid_data = preprocess_default(valid_data, just_alpha, min_length, lemmatize)
          
            test_data_preprocessed, test_data = preprocess_default(test_data, just_alpha, min_length, lemmatize)
        
        
        else:
            no_token = True
            import numpy as np
            import pickle
            
            train_data = np.load(os.path.join(data_directory, 'train.txt.npy'), allow_pickle = True, encoding = 'bytes')
            valid_data = np.load(os.path.join(data_directory, 'valid.txt.npy'), allow_pickle = True, encoding = 'bytes')
            test_data = np.load(os.path.join(data_directory, 'test.txt.npy'), allow_pickle = True, encoding = 'bytes')
            vocab = os.path.join(data_directory, 'vocab.pkl')
            vocab = pickle.load(open(vocab, 'rb'))
          
            print('Converting indexes back to words')
            idx2word = {v:k for k,v in vocab.items()}
            train_data_preprocessed = [[idx2word[x] for x in s] for s in train_data]
            # valid_data_preprocessed = [[idx2word[x] for x in s] for s in valid_data]
            test_data_preprocessed = [[idx2word[x] for x in s] for s in test_data]
            valid_data_preprocessed = test_data_preprocessed[-len(test_data)//20:]
            test_data_preprocessed = test_data_preprocessed[:len(test_data)-len(test_data)//20]
    
    else:
        incorrect_data_structure_msg = """The data structure for using custom dataset is incorrect, you should include the data argument to point to your custom directory. Inside that directory you should have three sub-directories named "train", "valid" and "test" each containing the training, validation and test set in respectively. Specifically, each sub-directory should include each input document in a different text file.'"""
        assert os.path.exists(os.path.join(data_directory, 'train')), print(incorrect_data_structure_msg)
        assert os.path.exists(os.path.join(data_directory, 'valid')), print(incorrect_data_structure_msg)
        assert os.path.exists(os.path.join(data_directory, 'test')), print(incorrect_data_structure_msg)
        
        no_token = False
        
        train_data = []
        for root, _, files in os.walk(os.path.join(data_directory, 'train')):
            for file in files:
                with open(os.path.join(root, file)) as f:
                    train_data.append(' '.join(f.readlines()))
        
        valid_data = []           
        for root, _, files in os.walk(os.path.join(data_directory, 'valid')):
            for file in files:
                with open(os.path.join(root, file)) as f:
                    valid_data.append(' '.join(f.readlines()))
                    
        test_data = []
        for root, _, files in os.walk(os.path.join(data_directory, 'test')):
            for file in files:
                with open(os.path.join(root, file)) as f:
                    test_data.append(' '.join(f.readlines()))
        
        if preprocess:
            train_data_preprocessed, train_data = preprocess_default(train_data, just_alpha, min_length, lemmatize)
          
            valid_data_preprocessed, valid_data = preprocess_default(valid_data, just_alpha, min_length, lemmatize)
          
            test_data_preprocessed, test_data = preprocess_default(test_data, just_alpha, min_length, lemmatize)
        
        else:
            train_data_preprocessed = train_data
          
            valid_data_preprocessed = valid_data
          
            test_data_preprocessed = test_data
    
    cv = CountVectorizer(max_features = max_target)
    
    if preprocess:
        Y_train = cv.fit_transform(train_data_preprocessed).toarray()
        Y_valid = cv.transform(valid_data_preprocessed).toarray()
        Y_test = cv.transform(test_data_preprocessed).toarray()
    else:
        Y_train = cv.fit_transform([' '.join(x) for x in train_data_preprocessed]).toarray()
        Y_valid = cv.transform([' '.join(x) for x in valid_data_preprocessed]).toarray()
        Y_test = cv.transform([' '.join(x) for x in test_data_preprocessed]).toarray()
    
    if use_glove:
        wordToIndex,indexToWord,wordToGlove=readGloveFile(glove_file)
        
        train_dataset = WordsDataset(train_data_preprocessed, Y_train, wordToIndex, no_token)
        valid_dataset = WordsDataset(valid_data_preprocessed, Y_valid, wordToIndex, no_token)
        test_dataset = WordsDataset(test_data_preprocessed, Y_test, wordToIndex, no_token)
        
        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn = train_dataset.collater)
        valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False, collate_fn = valid_dataset.collater)
        test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, collate_fn = test_dataset.collater)
        
        return train_dataloader, valid_dataloader, test_dataloader, wordToGlove, wordToIndex, cv.vocabulary_
        
    else:
        assert sentence_encoder is not None, 'You need to provide a SentenceTransformer object to create a sentence level VQ-VAE'
        
        if use_original:
            train_embeddings = encoder.encode(train_data)
            valid_embeddings = encoder.encode(valid_data)
            test_embeddings = encoder.encode(test_data)
        else:
            train_embeddings = encoder.encode([' '.join(x) for x in train_data_preprocessed])
            valid_embeddings = encoder.encode([' '.join(x) for x in valid_data_preprocessed])
            test_embeddings = encoder.encode([' '.join(x) for x in test_data_preprocessed])
            
        train_dataset = TensorDataset(train_embeddings, Y_train)
        valid_dataset = TensorDataset(valid_embeddings, Y_valid)
        test_dataset = TensorDataset(test_embeddings, Y_test)
        
        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False)
        test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
        
        
        return train_dataloader, valid_dataloader, test_dataloader, None, None, cv.vocabulary_
            
        
