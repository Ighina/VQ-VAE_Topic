#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 16:50:12 2022

@author: lacopoginhassi
"""
import re
import os
import sys
import json

import argparse
from modules import VQVAE
from load_data import load_data_pred
from sentence_transformers import SentenceTransformer
from pytorch_lightning import Trainer, seed_everything
import numpy as np
import torch

"""
TODO:
    1) CHANGE THE run.py script to output experiment folders for each experiment containing pretrained-model, topic vectors, topic matrix and results all in one place.
    2) IN THE same experiment folder above, change the run.py script to also save a dictionary as json containing all the hyperparameters.
    3a) IN THE json above, include an extra argument defining topic vectors path.
    3b) IN THE run.py script, change to always save topic vectors and include the path as above
    
"""

class Predictor():
    
    def __init__(self, args):
        if isinstance(args, dict):
            for arg in args:
                setattr(self, arg, args[arg])
            
            if 'topic_vector_file' not in args:
                self.topic_vector_file = None
            
        elif isinstance(args, argparse.Namespace):
            for arg in vars(args):
                setattr(self, arg, args[arg])
                
        
    
    def predict(self, return_topic_vectors = False):
    
        sentence_encoder = not self.glove
        
        if sentence_encoder:
            encoder = SentenceTransformer(self.encoder_model)
        else:
            encoder = None
        
        predict_loader, wordToGlove, wordToIndex = load_data_pred(self.glove, 
                                                                self.preprocess,
                                                                self.data,
                                                                 # self.dataset, DELETE THIS FROM load_data_pred 
                                                                 encoder,
                                                                 self.just_alpha,
                                                                 self.minimum_word_length,
                                                                 self.target_size,
                                                                 self.lemmatize,
                                                                 self.batch_size,
                                                                 self.glove_file)
        
        print(f'Length of prediction loader {len(predict_loader)}')
        
        glove_dim = int(re.findall('(\d+)d', self.glove_file)[0])
        
        emb_dim = glove_dim if self.glove else encoder.get_sentence_embedding_dimension()
        
        seed_everything(self.seed, workers=True)
        # sets seeds for numpy, torch and python.random.
        
        if self.multi_view:
            self.multi_head = True
        
        model = VQVAE.load_from_checkpoint(self.best_model_path,
                                           vocab_size = self.target_size,
                                           embedding_size = emb_dim,
                                           topics = self.topics,
                                           decoder_layers = self.decoder_layers,
                                           structural_encoder = self.encoder_layers>0,
                                           encoder_layers =   self.encoder_layers,
                                           sentence_encoder = sentence_encoder,
                                           lr = self.learning_rate,
                                           beta = self.beta,
                                           zeta = self.zeta,
                                           wordToGlove = wordToGlove,
                                           wordToIndex = wordToIndex,
                                           soft = self.soft,
                                           multi_head = self.multi_head,
                                           heads = self.heads,
                                           multi_view = self.multi_view)
        
        accelerator = 'gpu' if self.num_gpus >0 else 'cpu'
        
        trainer = Trainer(deterministic=True, 
                            accelerator = accelerator, 
                            gpus = self.num_gpus)
                            
                            
        trainer.predict(model, predict_loader)
        
        results = torch.stack(model.predictions).detach().cpu().numpy()
        
        if return_topic_vectors:
            assert self.topic_vector_file is not None
            top_vecs = np.load(self.topic_vector_file)
            
            doc_vecs = []
            for result in results:
                doc_vecs.append(np.matmul(top_vecs.T, result).reshape(-1))
                
            return results, np.array(doc_vecs)
        
        return results
        
        
if __name__=='__main__':
    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)
            
    
    parser = MyParser(
                description = 'Run training with parameters defined in the relative json file')
    
    parser.add_argument('--data', '-d', default = 'inputs', help = 'Input data directory')
    
    parser.add_argument('--out_directory', '-out', default = 'outputs', help = 'Output directory')
    
    parser.add_argument('--best_model_path', '-model', type = str, help = 'Path to model checkpoint')
    
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
    
    parser.add_argument('--topic_vector_file', type = str, help = 'The path to the saved topic vectors.')
    
    parser.add_argument('--hyperparameters', '-hp', type = str, help = 'Path to the hyperparameters json. If included, the sved hyperparameters will overwrite most of the arguments above.')
    
    parser.add_argument('--return_topic_vectors', '-rtv', action = 'store_true', help = 'Whether to also returned the dense representation of each input document as the weighted average of the topic vectors')
    
    args = parser.parse_args()
    
    rtv = args.return_topic_vectors
    out = args.out_directory
    
    if not os.path.exists(out):
        os.mkdir(out)
    
    if args.hyperparameters is not None:
        
        data = args.data
        gpus = args.num_gpus
        
        
        with open(args.hyperparameters) as f:
            args = json.load(f)
        
        args['num_gpus'] = gpus
        args['data'] = data
    else:
        assert args.best_model_path is not None, 'If not using pre-saved hyperparameters, give the path for the model checkpoint to be used.'
        
    model = Predictor(args)
    
    if rtv:
        results, doc_vecs = model.predict(return_topic_vectors=True)
        
        np.save(os.path.join(out, 'document_vectors'), doc_vecs)
        
    else:
        results = model.predict()
        
    np.save(os.path.join(out, 'probabilities'), results)
    