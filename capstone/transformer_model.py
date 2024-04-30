#https://nlp.seas.harvard.edu/annotated-transformer/
#https://www.youtube.com/watch?v=eMlx5fFNoYc&vl=en

#suppress warnings
#import torchtext; torchtext.disable_torchtext_deprecation_warning()

#IMPORTS, ORGNAIZE INTO CATEGORIES WHEN DONE
import concurrent.futures
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import spacy

import multiprocessing

import spacy
import en_core_web_md

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

import csv
import numpy as np



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



def attention(query, key, dropout=None):
    '''
        Computes value matrix result of scaled dot product attention of a query and key vector of two word vectors

        Args:
            query (FloatTensor): query vector (W_q * V_i)
            key (FloatTensor): key vector vector (W_k * V_i)

        Returns:
            Weighted value matrix of key word to add to query word
    '''
    
    d_k = query.size()[0]
    score = torch.dot(query, key) / math.sqrt(d_k)

    if(dropout is not None):
        score = dropout(score)
    
    return score



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
            context_window (int): size of the attention's context window, i.e. if context_window is x, the window will be x length to the left and right
            dropout: dropout rate of "neurons" in attention head

        Methods:
            __init__ (self, heads, d_embedding, dropout=0.1): constructor for object
            forward(self, input): forward pass of multi headed attention block, executes each head of attention
    '''

    def __init__ (self, heads, d_embedding, context_window, dropout=0.1):
        '''
            Basic constructor, defines all attributes

            Args:
                heads (int): number of attention heads in block
                d_embedding (int): dimension/number of coordinates in each word embedding
                context_window (int): size of the attention's context window, i.e. if context_window is x, the window will be x length to the left and right
                dropout: dropout rate of "neurons" in attention head

            Returns:
                None
        '''
        super(multi_head_attn, self).__init__()
        self.heads = heads 
        self.d_embedding = d_embedding
        self.context_window = context_window
        self.query_tensor = nn.Parameter(torch.randn(heads, d_embedding//heads, d_embedding)) #d_embedding//heads for now cause it seems like thats what ppl use
        self.key_tensor = nn.Parameter(torch.randn(heads, d_embedding//heads, d_embedding))
        self.value_down_tensor = nn.Parameter(torch.randn(heads, d_embedding//heads, d_embedding)) 
        self.value_up_tensor = nn.Parameter(torch.randn(heads, d_embedding, d_embedding//heads)) 
        self.dropout = nn.Dropout(p=dropout)
    

    def forward(self, input, training):
        '''
            Forward pass of multiheaded attention block, performs each head of attention

            Args:
                input (FloatTensor): list of word embeddings in input
            
            Returns:
                None, modifies input embeddings directly
        '''

        #apply attention for each head
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.heads) as executor:
            for i in range(self.heads):
                input += executor.submit(self.attn_block, input, i, training).result()

        input /= self.heads #avg of modified matrices


    def attn_block(self, input, i, training):
        '''
        '''

        #loop through each embedding for queries
        for j, E_q in enumerate(input):
            modified_input = input.clone()

            #i assume because multiplying by a vector with requires grad makes it require grad, and we dont want E_q requiring grad or else mult throws error, so need to make copy with no require grad
            with torch.no_grad(): 
                E_q_clone = E_q.clone().detach()

            query = torch.mv(self.query_tensor[i], E_q_clone) 
            delta_value = torch.zeros(self.d_embedding//self.heads)
            scores = []
            values = []


            #loop through each embedding in context window for keys
            l, r = max(j - self.context_window, 0), min(j + self.context_window, len(modified_input))
            for k in range(l, r):
                E_k = modified_input[k]

                key = torch.mv(self.key_tensor[i], E_k) #mv requires m to come first then v in args
                values.append(torch.mv(self.value_down_tensor[i], E_k))

                scores.append(attention(query, key, dropout=self.dropout if training else None))
            

            scores = torch.FloatTensor(scores)
            weights = F.softmax(scores, dim=0)

            for k in range(len(values)):
                delta_value += values[k]*weights[k]


            modified_input[j] += torch.mv(self.value_up_tensor[i], delta_value)
            
            modified_input[j] /= torch.mean(modified_input[j]) #scale down vectors, prevent inflation/deflation of vector magnitudes from constantly adding and multiplying
            modified_input[j] = (modified_input[j] - torch.mean(modified_input[j])) * (1.0 / torch.std(modified_input[j])) + torch.mean(modified_input[j]) #modify standard deviation, POTENTIALLY TWEAK VALUES FOR BETTER RESULTS

        #returns rather than directly modifies because with directly modifying, heads don't run perfectly in sync making the inputs inflate/deflate before they can be scaled down
        return modified_input




class feed_forward(nn.Module):
    '''
    
    '''

    def __init__(self, d_output, d_embedding, dropout=0.1):
        super(feed_forward, self).__init__()
        self.unembedding = nn.Parameter(torch.randn(d_output, d_embedding))
        self.dropout = nn.Dropout(dropout)


    def forward(self, input, training):
        if(training):
            self.unembedding = nn.Parameter(self.dropout(self.unembedding))
        
        return F.softmax(torch.mv(self.unembedding, input), dim=0)



class transformer(nn.Module):
    '''
    
    '''

    def __init__(self, d_output, d_embedding, context_window, heads):
        super(transformer, self).__init__()
        self.ff = feed_forward(d_output, d_embedding)
        self.attn = multi_head_attn(heads, d_embedding, context_window)
        self.gen_pe = gen_pe
    
    
    def predict(self, input, training):
        '''
        '''

        #add positional encoding
        d_words = input.size()[0]
        d_embedding = input.size()[1]

        input += self.gen_pe(d_words, d_embedding)


        #plug into attention then fnn
        self.attn(input, training)
        return self.ff(input[-1], training)



def stem(phrase):
    '''
    '''
    stemmer = PorterStemmer()
    
    words = word_tokenize(phrase)
    stop_words = stopwords.words("english")
    stemmed_words = []

    for word in words:
        if word not in stop_words: 
            stemmed_words.append(stemmer.stem(word))

    return " ".join(stemmed_words)


    

#driver code
d_output = 2
d_embedding = 300
context_window = 100
heads = 8
num_epochs = 5

model = transformer(d_output, d_embedding, context_window, heads) #num outputs, embedding size, num heads
training = True


X_train = []
y_train = []


#read csv file and append post descriptions to training data
with open("yrdsb_instagram_posts.csv", encoding="utf8") as csvfile:
    yrdsb_instagram_posts = csv.reader(csvfile)
    for row in yrdsb_instagram_posts:
        if len(row) != 0:
            X_train.append(stem(row[0]))
            y_train.append(int(row[1]))



#transform training data to word vectors
#tokenize each post
nlp = spacy.load("en_core_web_md") #tokenizer
docs = [nlp(nxt) for nxt in X_train]

X_train = []
#for every post
for i, doc in enumerate(docs):
    X_train.append([])

    #change token to vector representation
    for token in doc:
        X_train[i].append(token.vector)



#training
#define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-7)


for epoch in range(num_epochs):
    print(f"EPOCH: {epoch+1}/{num_epochs}")
    print("    Posts Trained:")

    for i, input in enumerate(X_train):
        optimizer.zero_grad()
        output = model.predict(torch.tensor(np.array(input)), training)[0] #0 is prob that it is snow day, 1 is prob that it isnt

        #criterion requires inputs to be tensors
        output_tensor = output.clone().detach().requires_grad_(True)
        expected_tensor = torch.tensor(y_train[i], dtype=torch.float32, requires_grad=True)

        loss = criterion(output_tensor, expected_tensor)
        loss.backward()
        optimizer.step()
        print(f"    {i+1}/{len(X_train)}, loss: {loss}")



stemmed_phrase = stem("Due to anticipated inclement weather not cancelled but it is an inclement weather day today so dont come to school")
test_doc = nlp(stemmed_phrase)
test_x = [token.vector for token in test_doc]

print(model.predict(torch.tensor(np.array(test_x)), False))
