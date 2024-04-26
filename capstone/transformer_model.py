#https://nlp.seas.harvard.edu/annotated-transformer/

import torchtext; torchtext.disable_torchtext_deprecation_warning()

import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP



def gen_pe(d_words, d_embedding):
    '''
        Generates positional encoding matrix to add to input matrix 

        Args:
            d_words (int): number of words for input
            d_embedding (int): number of coordinates in each word embedding
        
        Returns:
            Positional encoding matrix; matrix with sin and cos values in each position to give positional information to word vectors
    '''

    pe = torch.zeros(d_words, d_embedding)

    for i in range(d_words):
        for j in range(d_embedding//2):
            theta = i / (10000 ** ((2*i)/d_embedding))
            pe[i, 2*j] = math.sin(theta)
            pe[i, 2*j+1] = math.cos(theta)

    return pe



def attention(query, key, value, dropout=None):
    '''
        Computes value matrix result of scaled dot product attention of a query and key vector of two word vectors

        Args:
            query (LongTensor): query vector (W_q * V_i)
            key (LongTensor): key vector vector (W_k * V_i)
            vector (LongTensor): query vector (W_q * V_i)

        Returns:
            Weighted value matrix of key word to add to query word
    '''
    
    d_k = query.size()
    score = torch.matmul(query, key) / math.sqrt(d_k)
    weight = score.softmax()

    if(dropout is not None):
        weight = dropout(weight)
    
    return torch.matmul(value, weight)



class multi_head_attn(nn.Module):
    '''
        Multiheaded attention block. Combines attention heads of each query, key and value matrix into one block which influences word embeddings.

        Attributes:
            heads (int): number of attention heads in block
            d_embedding (int): dimension/number of coordinates in each word embedding
            query_tensor: tensor of all query matrices
            key_tensor: tensor of all key matrices
            value_down_tensor: tensor of all value_down matrices, value_down matrix is the factor of value matrix which is immediately added to embedding value
            value_up_tensor: tensor of all value_up matrices, value_up is the factor of value matrix which is multiplied by embeddings to give final word embedding
            dropout: dropout rate of "neurons" in attention head

        Methods:
            __init__ (self, heads, d_embedding, dropout=0.1): constructor for object
            forward(self, input): forward pass of multi headed attention block, executes each head of attention
    '''

    def __init__ (self, heads, d_embedding, dropout=0.1):
        '''
            Basic constructor, defines all attributes

            Args:
                heads (int): number of attention heads in block
                d_embedding (int): dimension/number of coordinates in each word embedding
                dropout: dropout rate of "neurons" in attention head

            Returns:
                None
        '''
        super(multi_head_attn, self).__init__()
        self.heads = heads 
        self.d_embedding = d_embedding
        self.query_tensor = nn.Parameter(torch.randn(heads, d_embedding//heads, d_embedding)) #d_embedding//heads for now cause it seems like thats what ppl use
        self.key_tensor = nn.Parameter(torch.randn(heads, d_embedding//heads, d_embedding))
        self.value_down_tensor = nn.Parameter(torch.randn(heads, d_embedding//heads, d_embedding)) 
        self.value_up_tensor = nn.Parameter(torch.randn(heads, d_embedding, d_embedding//heads)) 
        self.dropout = nn.Dropout(p=dropout)
    

    def forward(self, input):
        '''
            Forward pass of multiheaded attention block, performs each head of attention

            Args:
                input (LongTensor): list of word embeddings in input
            
            Returns:
                None, modifies input embeddings directly
        '''
        #apply attention for each head, NOT CURRENTLY RUN IN PARALLEL
        for i in range(self.heads):
            
            #loop through each embedding in context window for queries
            for j, E_q in enumerate(input):
                query = torch.matmul(self.query_tensor[i], E_q)
                delta_value = torch.zeros(self.d_embedding//self.heads) 

                #loop through each embedding in context window for keys
                for E_k in input:
                    key = torch.matmul(self.key_tensor[i], E_k)
                    value = torch.matmul(self.value_down_tensor[i], E_k)

                    weighted_value = attention(query, key, value, dropout=self.dropout)
                    delta_value += weighted_value
                

                input[j] += torch.matmul(delta_value, self.value_up_tensor[i])



class feed_forward(nn.Module):
    '''
    
    '''

    def __init__(self, d_vocab, d_embedding, dropout=0.1):
        super(feed_forward, self).__init__()
        self.unembedding = nn.Parameter(torch.randn(d_vocab, d_embedding))
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, training):
        if(training):
            input = self.dropout(input)
        
        return torch.matmul(input, self.unembedding).softmax()


    



print(gen_pe(10, 10))