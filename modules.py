#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 11:53:59 2022

@author: lacopoginhassi
"""

"""
Code mainly obtained from:
https://github.com/ritheshkumar95/pytorch-vqvae
"""
import torch
from torch.autograd import Function
import torch.nn as nn
from collections import OrderedDict
import pytorch_lightning as pl

class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)

def createPreTrainedEmbedding(word2Glove, word2index, isTrainable):
  vocab_size = len(word2index)+1
  embed_size = next(iter(word2Glove.values())).shape[0]
  matrix = torch.zeros((vocab_size, embed_size))
  for word, index in word2index.items():
    matrix[index,:] = torch.FloatTensor(word2Glove[word])

  embedding = nn.Embedding.from_pretrained(matrix)
  if not isTrainable:
    embedding.requires_grad = False
  return embedding

vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)

class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        # z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x, indices = vq_st(z_e_x, self.embedding.weight.detach())
        # z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x_bar = z_q_x_bar_flatten.view_as(z_e_x)
        # z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar, indices

class FakeEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.weight = nn.Parameter(torch.randn((K, D)))
        self.weight.data.uniform_(-1./D, 1./D)
    

class VQAttention(nn.Module):
    def __init__(self, K, D, multi_head = False, heads = 4, original = False, distance_attention = False):
        super().__init__()
        self.topics = K
        self.embedding = FakeEmbedding(K, D)
        
        self.soft = nn.Softmax(dim=2)
        self.mh = multi_head
        if multi_head:
            if original:
                self.original = True
                self.view_project = nn.Parameter(torch.randn(heads, D, D))
                self.proj_back = nn.Linear(heads*D, D)
                self.view_activation = nn.Tanh()
            else:
                self.original = False
                self.attention = nn.MultiheadAttention(D, heads, batch_first=True)
        
        self.distance_attention = distance_attention
    
    def multiview_attention(self, x, original = False):
        """
        If original is True, as described in the original paper, but the implementation does not seem to lead to good results... 
        our original implementation is the default.
        """
        
        batch, seq, emb = x.shape
        
        x = x.contiguous().view(-1, emb) # (batch*seq, D)
        
        multi_code = torch.matmul(self.view_project, self.embedding.weight.T) # (heads, D, K)
        
        U = self.soft(torch.matmul(x, multi_code)) # (head, batch*seq, K)
        
        Q = torch.matmul(U, self.embedding.weight) # (head, batch*seq, D)
        
        zqn = Q.permute(2,1,0).contiguous().view(batch*seq, -1) # (batch, seq, D*head)
        
        z_q_x = self.view_activation(self.proj_back(zqn)).view(batch, seq, -1) # (batch, seq, D)
        
        return z_q_x, torch.sum(U, axis = 0).contiguous().view(batch, seq, -1)
       
    def forward(self, z_e_x):
        batch, seq, emb = z_e_x.shape
        
        if self.mh:
            if self.original:
                z_q_x, attention_scores = self.multiview_attention(z_e_x)
            else:
                batched_embedding = self.embedding.weight.expand(batch, self.topics, emb)
                z_q_x, attention_scores = self.attention(z_e_x, batched_embedding, batched_embedding)
            
        elif self.distance_attention:
            """
            Using euclidean distance to compute attentions instead of dot product. It leads to comparable results.
            """
            
            z_e_x_ = z_e_x.view(-1, emb)
            
            codebook_sqr = torch.sum(self.embedding.weight ** 2, dim=1)
            inputs_sqr = torch.sum(z_e_x_ ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                z_e_x_, self.embedding.weight.t(), alpha=-2.0, beta=1.0)
            
            distances = distances.view(batch, seq, -1)
            
            attention_scores = self.soft(distances)
            
            z_q_x = torch.matmul(attention_scores, self.embedding.weight)
        else:
            """Traditional dot-product attention"""
            
            attention_scores = self.soft(torch.matmul(z_e_x, self.embedding.weight.T))
            
            z_q_x = torch.matmul(attention_scores, self.embedding.weight)
            
        return z_q_x, attention_scores

class VectorQuantizedVAE(nn.Module):
    def __init__(self, vocab_size, embedding_size, topics, K=256,
                 structural_encoder = False,
                 sentence_encoder = False,
                 activation = nn.ReLU,
                 encoder_layers = 2,
                 hidden_dim = 512,
                 decoder_layers = 0,
                 wordToGlove = None,
                 wordToIndex = None,
                 soft = False,
                 multi_head = False,
                 heads = 2,
                 multi_view = False):
        super().__init__()
        
        self.sentence_encoder = sentence_encoder
        if not self.sentence_encoder:
          
          assert wordToGlove is not None, 'The dictionary containing the pre-trained embeddings needs to be provided to the argument wordToGlove!'
          assert wordToIndex is not None, 'The dictionary containing the word to index mapping of the pre-trained embeddings needs to be provided to the argument wordToGlove!'
          
          self.embedding = createPreTrainedEmbedding(wordToGlove, wordToIndex, False)

        if structural_encoder:
          self.structural_encoder = True
          enc_list = OrderedDict([
                ('embedding2hidden', nn.Linear(embedding_size, hidden_dim)),
                ('activation', activation())])
            
          for i in range(encoder_layers-2): 
              enc_list['linear'+str(i)] = nn.Linear(hidden_dim, hidden_dim)
              enc_list['activation'+str(i)] = activation()

          enc_list['hidden2latent'] = nn.Linear(hidden_dim, K)
          enc_list['latent_activation'] = activation()
            
          self.encoder = nn.Sequential(enc_list)

        else:
          self.structural_encoder = False
          K = embedding_size

        
        if soft:
            self.soft = True
            self.codebook = VQAttention(topics, K, multi_head, heads, multi_view)
        else:
            self.soft = False
            self.codebook = VQEmbedding(topics, K)

        if decoder_layers>0:
          dec_list = OrderedDict([
              ('latent2hidden', nn.Linear(K, hidden_dim)),
              ('activation', activation())])
          
          for i in range(decoder_layers-2): 
              dec_list['linear'+str(i)] = nn.Linear(hidden_dim, hidden_dim)
              dec_list['activation'+str(i)] = activation()
          
          dec_list['hidden2output'] = nn.Linear(hidden_dim, vocab_size)
          dec_list['output_activation'] = nn.ReLU()

        else:
          dec_list = OrderedDict([
                                  ('topic2output', nn.Linear(K, vocab_size)),
                                  ('output_activation', nn.ReLU())
          ])
        
        # encoder, decoder
        self.decoder = nn.Sequential(dec_list)

        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        # z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(self.codebook.embedding[latents])
        return x_tilde

    def forward(self, x, lengths = None):
        if not self.sentence_encoder:
            x = self.embedding(x)

        if self.structural_encoder:

          z_e_x = self.encoder(x)

        else:

          z_e_x = x
        
        if self.soft:
            z_q_x, indices = self.codebook(z_e_x) # here indices are actually word-topic attention weights
        else:
            z_q_x_st, z_q_x, indices = self.codebook.straight_through(z_e_x)
        
        if not self.sentence_encoder:
            if lengths is not None:
                
                assert lengths.shape[0]==z_q_x.shape[0]
                
                batched_means = []
                for index, length in enumerate(lengths):
                    if length:
                        batched_means.append(torch.mean(z_q_x[index][:length], axis=0))
                    else:
                        raise ValueError("Empty Sentence in the Dataset!")
                
                z_q_x_st = torch.stack(batched_means)
            else:
                z_q_x_st = torch.mean(z_q_x, axis = 1)
        else:
          z_q_x_st = z_q_x

        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x, indices
    
    
class VQVAE(pl.LightningModule):
    def __init__(self, 
                 vocab_size,
                 embedding_size,
                 enc_out_dim=512, 
                 latent_dim=256, 
                 topics = 50,
                 activation=nn.ReLU,
                 encoder_layers = 2,
                 decoder_layers = 2,
                 structural_encoder = False,
                 sentence_encoder = False,
                 lr = 1e-4,
                 beta = 0.25,
                 zeta = 0.001,
                 wordToGlove=None,
                 wordToIndex=None,
                 soft = False,
                 multi_head = False,
                 heads = 2,
                 multi_view = False,
                 compute_perplexity = False):
      
        super().__init__()

        # self.save_hyperparameters()
        
        self.model = VectorQuantizedVAE(vocab_size,
                                        embedding_size,
                                        topics,
                                        latent_dim,
                                        structural_encoder,
                                        sentence_encoder,
                                        activation,
                                        encoder_layers,
                                        enc_out_dim,
                                        decoder_layers,
                                        wordToGlove,
                                        wordToIndex,
                                        soft,
                                        multi_head,
                                        heads,
                                        multi_view)
        
        self.mse_loss = nn.MSELoss()
        self.lts_loss = nn.HingeEmbeddingLoss()
        self.lts_target = torch.tensor([[1.0 if x==i else -1.0 for i in range(topics)] for x in range(topics)])

        self.lr = lr
        self.beta = beta
        self.zeta = zeta
        
        self.topic_clusters = {t:{} for t in range(topics)}
        self.perplexity = compute_perplexity

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def forward(self, batch):
        x_tilde, z_e_x, z_q_x, indices = self.model(batch)
        
        n_topics = self.model.codebook.embedding.weight.shape[0]
        
        if self.soft:
            soft = nn.Softmax(dim=1)
            probabilities = soft(indices.sum(axis = 1))
            
        else:
            indices = indices.contiguous().view(batch.shape[0], -1)
            
            probabilities = []
            
            for sequence in indices:
                probabilities.append(torch.bincount(sequence, minlength=n_topics)/len(sequence))
                
            probabilities = torch.stack(probabilities)
        
        return probabilities, z_q_x
    
    def training_step(self, batch, batch_idx):
        if self.model.sentence_encoder:
            x, y = batch
            lengths = None
        else:
            x, y, lengths = batch

        x_tilde, z_e_x, z_q_x, indices = self.model(x, lengths)

        # Reconstruction loss
        loss_recons = self.mse_loss(x_tilde, y)
        # Vector quantization objective
        loss_vq = self.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = self.mse_loss(z_e_x, z_q_x.detach())

        loss_lts = self.lts_loss(torch.cdist(self.model.codebook.embedding.weight,
                                        self.model.codebook.embedding.weight),
                            self.lts_target.to(x_tilde.device))

        loss = loss_recons + loss_vq + self.beta * loss_commit + self.zeta * loss_lts
        
        self.log_dict({
            'lts_loss': loss_lts,
            'commitment_loss': loss_commit,
            'recon_loss': loss_recons,
            'loss': loss
        })

        return loss

    def validation_step(self, batch, batch_idx):
        if self.model.sentence_encoder:
            x, y = batch
            lengths = None
        else:
            x, y, lengths = batch

        x_tilde, z_e_x, z_q_x, indices = self.model(x, lengths)

        # Reconstruction loss
        valid_loss_recons = self.mse_loss(x_tilde, y)
        # Vector quantization objective
        valid_loss_vq = self.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        valid_loss_commit = self.mse_loss(z_e_x, z_q_x.detach())

        valid_loss_lts = self.lts_loss(torch.cdist(self.model.codebook.embedding.weight,
                                        self.model.codebook.embedding.weight),
                            self.lts_target.to(x_tilde.device))

        valid_loss = valid_loss_recons + valid_loss_vq + self.beta * valid_loss_commit + self.zeta * valid_loss_lts
        
        self.log_dict({
            'valid_lts_loss': valid_loss_lts,
            'valid_commitment_loss': valid_loss_commit,
            'valid_recon_loss': valid_loss_recons,
            'valid_loss': valid_loss
        }, on_epoch = True, prog_bar=True)

        return valid_loss  
    
    
    def test_step(self, batch, batch_idx):
        if self.model.sentence_encoder:
            x, y = batch
            lengths = None
        else:
            x, y, lengths = batch

        x_tilde, z_e_x, z_q_x, indices = self.model(x, lengths)

        # Reconstruction loss
        test_loss_recons = self.mse_loss(x_tilde, y)
        # Vector quantization objective
        test_loss_vq = self.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        test_loss_commit = self.mse_loss(z_e_x, z_q_x.detach())

        test_loss_lts = self.lts_loss(torch.cdist(self.model.codebook.embedding.weight,
                                        self.model.codebook.embedding.weight),
                            self.lts_target.to(x_tilde.device))
        
        if self.perplexity:
            # log-exp trick to sum the probabilities of each word in the one-hot representation under each topic, following the topic matrix.
            perplexity = -torch.log(torch.exp(torch.matmul(y, torch.log(self.topic_matrix.T.to(y.device)))).sum(axis=1)).sum()/y.sum()
        
        if self.model.sentence_encoder:
          for idx, topic in enumerate(indices):
            topic = topic.detach().cpu().item()
            
            if topic not in self.topic_clusters:
              self.topic_clusters[topic] = {}

            for word_idx, word in enumerate(y[idx]):
              if word.detach().cpu().item()>0:
                if word_idx not in self.topic_clusters[topic]:
                  self.topic_clusters[topic][word_idx] = 1
                else:
                  self.topic_clusters[topic][word_idx] += 1

        test_loss = test_loss_recons + test_loss_vq + self.beta * test_loss_commit + self.zeta * test_loss_lts
        
        if self.perplexity:
            self.log_dict({
                'lts_loss': test_loss_lts,
                'commitment_loss': test_loss_commit,
                'recon_loss': test_loss_recons,
                'loss': test_loss,
                'perplexity': perplexity
            }, on_epoch = True, prog_bar=True)
        else:
            self.log_dict({
                'lts_loss': test_loss_lts,
                'commitment_loss': test_loss_commit,
                'recon_loss': test_loss_recons,
                'loss': test_loss
            }, on_epoch = True, prog_bar=True)
            

        return test_loss