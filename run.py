#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 16:50:12 2022

@author: lacopoginhassi
"""
import os
import sys

import argparse
from modules import VQVAE
from utils import top_k_topic, print_all_topics, class_tfidf
from load_data import load_data
from sentence_transformers import SentenceTransformer
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
import torch

import requests
import re
import json

def main(args):
    
    sentence_encoder = not args.glove
    
    if sentence_encoder:
        encoder = SentenceTransformer(args.encoder_model)
    else:
        encoder = None
    
    train_loader, valid_loader, test_loader, wordToGlove, wordToIndex, target_vocab = load_data(args.glove, 
                                                                                  args.preprocess,
                                                                                  args.data,
                                                                                  args.dataset,
                                                                                  encoder,
                                                                                  args.just_alpha,
                                                                                  args.minimum_word_length,
                                                                                  args.target_size,
                                                                                  args.lemmatize,
                                                                                  args.batch_size,
                                                                                  args.glove_file)
    
    print(f'Length training set {len(train_loader)}')
    print(f'Length validation set {len(valid_loader)}')
    print(f'Length test set {len(test_loader)}')
    
    out = os.path.join(args.out_directory, args.experiment_name)
    
    while os.path.exists(out):
        out += '0'
    
    os.mkdir(out)
    
    glove_dim = int(re.findall('(\d+)d', args.glove_file)[0])
    
    emb_dim = glove_dim if args.glove else encoder.get_sentence_embedding_dimension()
    
    seed_everything(args.seed, workers=True)
    # sets seeds for numpy, torch and python.random.
    
    if args.multi_view:
        args.multi_head = True
    
    model = VQVAE(vocab_size = len(target_vocab),
                    embedding_size = emb_dim,
                     topics = args.topics,
                     decoder_layers = args.decoder_layers,
                     structural_encoder = args.encoder_layers>0,
                     encoder_layers = args.encoder_layers,
                     sentence_encoder = sentence_encoder,
                     lr = args.learning_rate,
                     beta = args.beta,
                     zeta = args.zeta,
                     wordToGlove = wordToGlove,
                     wordToIndex = wordToIndex,
                    soft = args.soft,
                    multi_head = args.multi_head,
                    heads = args.heads,
                    multi_view = args.multi_view)
    
    accelerator = 'gpu' if args.num_gpus >0 else 'cpu'
    
    checkpoint_callback = ModelCheckpoint(
                monitor='valid_loss',
                dirpath= out,
                filename='checkpoint-{epoch:02d}-{valid_loss:.2f}',
                save_top_k=1,
                mode='min',
            )
    
    
    trainer = Trainer(callbacks = [checkpoint_callback], 
                        deterministic=True, 
                        accelerator = accelerator, 
                        gpus = args.num_gpus, 
                        max_epochs = args.max_epochs,
                        auto_lr_find = args.auto_lr)
                        
                        
    if args.auto_lr:
        trainer.tune(model, train_loader, valid_loader)
    
    trainer.fit(model, train_loader, valid_loader)
    
    model = VQVAE.load_from_checkpoint(checkpoint_callback.best_model_path,
                                    vocab_size = len(target_vocab),
                                    embedding_size = emb_dim,
                                    topics = args.topics,
                                    decoder_layers = args.decoder_layers,
                                    structural_encoder = args.encoder_layers>0,
                                    encoder_layers = args.encoder_layers,
                                    sentence_encoder = sentence_encoder,
                                    lr = args.learning_rate,
                                    beta = args.beta,
                                    zeta = args.zeta,
                                    wordToGlove = wordToGlove,
                                    wordToIndex = wordToIndex,
                                    soft = args.soft,
                                    multi_head = args.multi_head,
                                    heads = args.heads,
                                    multi_view = args.multi_view,
                                    compute_perplexity = args.return_perplexity)
    
    topic_vectors = model.model.codebook.embedding.weight.detach().cpu().numpy()
        
    outfile = 'topic_vecs'
        
    np.save(os.path.join(out, outfile), topic_vectors)
    
    if sentence_encoder:
        trainer.test(model, train_loader)
        
        topic_matrix = class_tfidf(model.topic_clusters, target_vocab)
        
    else:
        topic_matrix_ = torch.matmul(model.model.codebook.embedding.weight, model.model.embedding.weight.T)
        
        topic_matrix = topic_matrix_.detach().cpu().numpy()
        
    
    if args.save_topic:
        
        if sentence_encoder:
            index2word = {v:k for k,v in target_vocab.items()}
            columns = [index2word(x) for x in range(len(target_vocab))]
        else:
            index2word = {v:k for k,v in wordToIndex.items()}
            columns = [index2word[x] if x>0 else 'pad_token' for x in range(len(index2word)+1)]
        top_df = pd.DataFrame(topic_matrix, columns = columns)
        
        outfile = 'topic_matrix'+'.csv'
        
        top_df.to_csv(os.path.join(out, outfile))
        
        
    if args.glove:
        topic_matrix = topic_matrix_
        
    if args.evaluate_on_target or args.return_perplexity:
        use_pytorch = False
        new_topic_matrix = (np.zeros((len(target_vocab), args.topics))+topic_matrix[:,2].detach().cpu().numpy()).T # unknown word index for all topics
        for topic in range(args.topics):
            index2word = {v:k for k,v in target_vocab.items()}
            for word, index in target_vocab.items():
                try:
                    new_topic_matrix[topic, index] = topic_matrix[topic, wordToIndex[word]]
                except KeyError:
                    pass
        
        topic_matrix = new_topic_matrix
    else:
        use_pytorch = args.glove
        
    print_all_topics(topic_matrix, index2word, k = args.topk, pt = use_pytorch)
    
    results = {}
    
    k=10
    metrics = ['cv', 'ca', 'cp', 'uci', 'npmi', 'umass', 'td']
    
    base_query = 'http://palmetto.aksw.org/palmetto-webapp/service/COHERENCE?words=WORDS'
        
    for metric in metrics:
      start = 0
      all_words = set()
      metric_query = re.sub('COHERENCE', metric, base_query)
      for i in range(topic_matrix.shape[0]):
        top_words = list(top_k_topic(i, topic_matrix, index2word, k, use_pytorch).keys())
        if metric=='td':
          all_words = all_words.union(set(top_words))
        else:
          topic_query = re.sub('WORDS', ' '.join(top_words), metric_query)
          response = requests.get(topic_query)
    
          if response.status_code!=400:
            start += float(response.text)
          else:
            raise ValueError()
    
      if len(all_words)>0:
        results[metric] = len(all_words)/(k*(i+1))
      else:
        results[metric] = start/(i+1) 
    
    results['ti'] = results['cv']*results['td']
    
    if args.return_perplexity:
        
        print("Computing perplexity on test set...")
        soft = torch.nn.Softmax(dim=1)
        model.topic_matrix = soft(torch.tensor(topic_matrix, dtype = torch.float))
        
        ppl = trainer.test(model, dataloaders = test_loader)[0]['perplexity']
        results['perplexity'] = np.exp(ppl)
    
    print(results)
    
    outfile = 'coherence_scores'+'.json'
    
    with open(os.path.join(out, outfile), 'w') as f:
        json.dump(results, f)
    
    hyperparameters = {}
    for arg in vars(args):     
        hyperparameters[arg] = getattr(args, arg)
    
    hyperparameters['topic_vector_file'] = os.path.join(out, 'topic_vecs.npy')
    hyperparameters['target_size'] = len(target_vocab)
    hyperparameters['best_model_path'] = checkpoint_callback.best_model_path
    
    with open(os.path.join(out, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparameters, f)
        
if __name__=='__main__':
    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)
            
    
    parser = MyParser(
                description = 'Run training with parameters defined in the relative json file')
    
    parser.add_argument('--data', default = 'data', help = 'Input data directory')
    
    parser.add_argument('--dataset', '-ds', default = 'ng20', help = 'Dataset to be used. Currently supported are ng20 and custom.')
    
    parser.add_argument('--out_directory', default = 'experiments', help = 'Output directory')
    
    parser.add_argument('--experiment_name', '-exp', default = 'exp0', type = str, help = 'Name of the current experiment: a folder with the same name will be created in the output directory in order to store all the results.')
    
    parser.add_argument('--glove', action= 'store_false', help = 'If included do not use glove but sentence encoders (not reccomended)')
    
    parser.add_argument('--glove_file', default = 'data/glove.6B.300d.txt', type = str, help = 'The GloVe file to load word embeddings.')
    
    parser.add_argument('--soft', '-s', action = 'store_true', help = 'If included, use soft VQ-VAE instead of hard VQ-VAE (see original paper)')
    
    parser.add_argument('--multi_head', '-mh', action = 'store_true', help = 'If included and using soft VQ-VAE, apply multihead attention instead of normal attention (similar to the multi-view model in the original paper)')
    
    parser.add_argument('--multi_view', '-mv', action = 'store_true', help = 'If included, use the multiview approach described in the original paper (different from standard multihead attention)')
    
    parser.add_argument('--heads', '-nh', default = 2, type = int, help = 'If using multihead attention, how many heads to use.')
    
    parser.add_argument('--save_topic', '-st', action='store_true', help = 'If included, it saves the topic vectors and matrices of the experiment in the output directory')
    
    parser.add_argument('--encoder_model', default = 'all-MiniLM-L12-v2', help = 'The sentence encoder to use if not using the default GloVe embeddings')
    
    parser.add_argument('--preprocess', '-p', action = 'store_true', help = 'If included, perform some minimal preprocessing, otherwise input data are assumed to be already preprocessed.')
    
    parser.add_argument('--topics', '-nt', default = 50, type = int, help = 'The number of topics in the model')
    
    parser.add_argument('--decoder_layers', default = 0, type = int, help = 'How many layers in the decoder. Default is None, as the original paper.')
    
    parser.add_argument('--encoder_layers', default = 0, type = int, help = 'How many layers in the encoder. Default is None, as the original paper.')
    
    parser.add_argument('--learning_rate', '-lr', default = 2e-4, type = float, help = 'The learning rate to be used in training.')
    
    parser.add_argument('--batch_size', '-bs', default = 50, type = int, help = 'The batch size to be used.')
    
    parser.add_argument('--beta', '-b', default = 0.25, type = float, help = 'The beta parameter weighting the commitment loss in the original paper')
    
    parser.add_argument('--zeta', '-z', default = 0.001, type = float, help = 'The zeta parameter weighting the latent topic space loss in the original paper')
    
    
    parser.add_argument('--num_gpus', '-gpu', default = 0, type = int, help = 'The number of gpus to use in training.')
    
    parser.add_argument('--max_epochs', '-ep', default = 100, type = int, help = 'The number of epochs for training the model.')
    
    parser.add_argument('--seed', default = 42, type = int, help = 'The random seed to replicate results')
    
    parser.add_argument('--auto_lr', '-al', action = 'store_true', help = 'If included, find the best learning rate with the automatic method suggested in pytorch lightning documentation.')
    
    parser.add_argument('--topk', default = 10, type = int, help = 'Top k words to be used in topic model evaluation.')
    
    parser.add_argument('--just_alpha', '-ja', action = 'store_true', help = 'If included and if applying preprocessing, delete all non-alphabetic characters (including numbers)')
    
    parser.add_argument('--minimum_word_length', '-mwl', default = 0, type = int, help = 'If included and if applying preprocessing, delete all words that are shorter or equal in length to the given value')
    
    parser.add_argument('--lemmatize', '-lemma', action = 'store_true', help = 'If included and if applying preprocessing, apply lemmatization to the text as part of the preprocessing.')
    
    parser.add_argument('--target_size', '-ts', default = 2000, type = int, help = 'The size of the output from the neural network, corresponding to the total number of words in the target vocabulary. The number of words is restricted by using the n most frequent ones.')
    
    parser.add_argument('--evaluate_on_target', '-eot', action = 'store_true', help = 'If included, this option restricts the size of the topic matrix to include just the words that are present in the target matrix. This is done to match the configuration of other topic models.')
    
    parser.add_argument('--return_perplexity', '-ppl', action = 'store_true', help = 'Whether to return the perplexity of the model on test documents.')
    
    args = parser.parse_args()
    
    main(args)
    
    
    
    
    
    