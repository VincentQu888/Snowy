#resources (thank you)
#https://nlp.seas.harvard.edu/annotated-transformer/
#https://www.youtube.com/watch?v=eMlx5fFNoYc&vl=en


#IMPORTS
#pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#spacy imports (english model)
import spacy
import en_core_web_md

#nltk imports (word stemming and tokenizing)
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

#multithreading import
import concurrent.futures

#other imports
import numpy as np
import math
import csv
import time

#import torchtext; torchtext.disable_torchtext_deprecation_warning()



#start of code:
def gen_pe(d_words, d_embedding):
    '''
        Generates positional encoding (pe) matrix to add to input matrix 

        Args:
            d_words (int): number of words for input
            d_embedding (int): number of coordinates in each word embedding

        Returns:
            Positional encoding matrix; matrix with sin and cos values in each position to give positional information to word vectors
    '''

    #generates base pe matrix
    pe = torch.zeros(d_words, d_embedding)

    #calculate pe for each index i,j
    for i in range(d_words):
        for j in range(d_embedding//2): #even and odd
            theta = i / (10000 ** ((2*i)/d_embedding))
            pe[i, 2*j] = math.sin(theta)
            pe[i, 2*j+1] = math.cos(theta) #sinusoidal pe

    return pe #returns pe to be added to input embeddings



def attention(query, key, dropout=None):
    '''
        Computes value matrix result of scaled dot product attention of a query and key vector of two word vectors

        Args:
            query (FloatTensor): query vector (W_q * V_i)
            key (FloatTensor): key vector vector (W_k * V_i)

        Returns:
            Weighted value matrix of key word to add to query word
    '''

    #compute dot product of query and key vectors
    d_k = query.size()[0]
    score = torch.dot(query, key) / math.sqrt(d_k)

    #apply dropout for training
    if(dropout is not None):
        score = dropout(score)

    return score #attention score to be normalized



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
            forward(self, input): forward pass of multi headed attention block, executes each head of attention in parallel
            attn_block(self, input, i, training): executes one attention head
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
                training (boolean): whether or not the prediction is used for training or not, enables dropout
            Returns:
                None, modifies input embeddings directly
        '''

        #apply attention for each head
        #NOTE: Uses multithreading/cpu parallelization rather than gpu since it's running on a school computer with integrated graphics, potential extra performance isn't worth it for this project training overnight anyways
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.heads) as executor:
            for i in range(self.heads):
                input += executor.submit(self.attn_block, input, i, training).result()

        input /= self.heads #avg of modified matrices


    def attn_block(self, input, i, training):
        '''
            Singular attention head, computes self scaled dot product attention of entire input matrix with query, key and value matrices i
    
            Args:
                input (FloatTensor): list of word embeddings in input
                i (int): index of query key and value matrix used for attention head
                training (boolean): whether or not the prediction is used for training or not, enables dropout
            Returns:
                modified_input (FloatTensor): modified input embeddings
        '''

        #loop through each embedding for queries
        for j, E_q in enumerate(input):
            modified_input = input.clone()

            #i assume because multiplying by a vector with requires grad makes it require grad, and we dont want E_q requiring grad or else mult throws error, so need to make copy with no require grad
            with torch.no_grad(): 
                E_q_clone = E_q.clone().detach() #word embedding for query

            query = torch.mv(self.query_tensor[i], E_q_clone) #create query vector
            delta_value = torch.zeros(self.d_embedding//self.heads) #delta_value to be added to input embeddings
            scores = [] #attn scores
            values = [] #value_down vectors for each embedding


            #loop through each embedding in context window for keys
            l, r = max(j - self.context_window, 0), min(j + self.context_window, len(modified_input)) #left and right boundary for context window
            for k in range(l, r):
                E_k = modified_input[k] #word embedding for key

                key = torch.mv(self.key_tensor[i], E_k) #mv requires m to come first then v in args
                values.append(torch.mv(self.value_down_tensor[i], E_k)) #value vector

                scores.append(attention(query, key, dropout=self.dropout if training else None)) #append score to attention scores


            scores = torch.FloatTensor(scores) #convert scores to tensor for softmax
            weights = F.softmax(scores, dim=0)

            #add weighted value vectors to delta_value
            for k in range(len(values)):
                delta_value += values[k]*weights[k]


            #add delta value to input
            modified_input[j] += torch.mv(self.value_up_tensor[i], delta_value)

            modified_input[j] /= torch.mean(modified_input[j]) #scale down vectors, prevent inflation/deflation of vector magnitudes from constantly adding and multiplying
            modified_input[j] = (modified_input[j] - torch.mean(modified_input[j])) * (1.0 / torch.std(modified_input[j])) + torch.mean(modified_input[j]) #modify standard deviation, POTENTIALLY TWEAK VALUES FOR BETTER RESULTS

        #returns rather than directly modifies because with directly modifying, heads don't run perfectly in sync making the inputs inflate/deflate before they can be scaled down
        return modified_input




class feed_forward(nn.Module):
    '''
        Feed forward neural network (fnn) for transformer

        Attributes:
            d_output (int): number of outputs for fnn, 2 for binary classification
            d_embedding (int): dimension/number of coordinates in each word embedding
           dropout: dropout rate of neurons in fnn

        Methods:
            __init__(self, d_output, d_embedding, dropout=0.1): constructor for object
            forward(self, input, training): forward pass of fnn
    '''

    def __init__(self, d_output, d_embedding, dropout=0.1):
        '''
            Basic constructor, defines all attributes

            Args:
                d_output (int): number of outputs for fnn, 2 for binary classification
                d_embedding (int): dimension/number of coordinates in each word embedding
                dropout: dropout rate of neurons in fnn

            Returns:
                None
        '''
        
        super(feed_forward, self).__init__()
        self.unembedding = nn.Parameter(torch.randn(d_output, d_embedding))
        self.dropout = nn.Dropout(dropout)


    def forward(self, input, training):
        '''
            Forward pass of fnn, performs neural network calculations and classifies results

            Args:
                input (FloatTensor): list of word embeddings in input)
                dropout: dropout rate of neurons in fnn

            Returns:
                Softmax of tensor output of fnn, tensor[0] = probability of snowday, tensor[1] = probability of not snowday
        '''

        #apply dropout if training
        if(training):
            self.unembedding = nn.Parameter(self.dropout(self.unembedding))

        #apply unembedding matrix
        return F.softmax(torch.mv(self.unembedding, input), dim=0)



class transformer(nn.Module):
    '''
        Whole transformer model, combines multiheaded attention, feed forward neural network and positional encoding

        Attributes:
            heads (int): number of attention heads in block
            d_embedding (int): dimension/number of coordinates in each word embedding
            d_output (int): number of outputs for fnn, 2 for binary classification
            context_window (int): size of the attention's context window, i.e. if context_window is x, the window will be x length to the left and right

        Methods:
            __init__(self, d_output, d_embedding, context_window, heads): constructor for object
            predict(self, input, training): Executes entire model architecture, multiheaded attention, feed forward neural network and positional encoding
    '''

    def __init__(self, d_output, d_embedding, context_window, heads):
        '''
            Basic constructor, defines all attributes

            Args:
                heads (int): number of attention heads in block
                d_embedding (int): dimension/number of coordinates in each word embedding
                d_output (int): number of outputs for fnn, 2 for binary classification
                context_window (int): size of the attention's context window, i.e. if context_window is x, the window will be x length to the left and right

            Returns:
                None
        '''
        
        super(transformer, self).__init__()
        self.ff = feed_forward(d_output, d_embedding)
        self.attn = multi_head_attn(heads, d_embedding, context_window)
        self.gen_pe = gen_pe


    def predict(self, input, training):
        '''
            Executes entire model architecture, multiheaded attention, feed forward neural network and positional encoding

            Args:
                input (FloatTensor): list of word embeddings in input
                training (boolean): whether or not the prediction is used for training or not, enables dropout

            Returns:
                Prediction tensor of fnn, tensor[0] = probability of snowday, tensor[1] = probability of not snowday
        '''

        #add positional encoding
        d_words = input.size()[0]
        d_embedding = input.size()[1]

        #modify input with positional encodings
        input += self.gen_pe(d_words, d_embedding)


        #plug sentence into attention then fnn
        self.attn(input, training)
        return self.ff(input[-1], training)



def stem(phrase):
    '''
        Stems a string phrase with nltk stemmer

        Args:
            phrase (string): string to stem

        Returns:
            stemmed_phrase (string): stemmed version of input string
    '''

    #define stemmer and stopwords
    stemmer = PorterStemmer()
    stop_words = stopwords.words("english")

    #tokenize phrase 
    words = word_tokenize(phrase)

    #add tokens to list of stemmed tokens
    stemmed_words = []
    for word in words:
        if word not in stop_words: #filter out stop words
            stemmed_words.append(stemmer.stem(word))

    return " ".join(stemmed_words) #join back list to string




#driver code
start_time = time.time()

d_output = 2
d_embedding = 300
context_window = 100
heads = 8
num_epochs = 5

model = transformer(d_output, d_embedding, context_window, heads) #num outputs, embedding size, num heads
training = True


#training data sets
X_train = [] #input
y_train = [] #expected output


#read csv file and append post descriptions to training data
with open("yrdsb_instagram_posts.csv", encoding="utf8") as csvfile:
    yrdsb_instagram_posts = csv.reader(csvfile) #read csv file and store in variable

    #for every post
    for row in yrdsb_instagram_posts:
        if len(row) != 0: #ensure row is not empty
            X_train.append(stem(row[0]))
            y_train.append(int(row[1]))



#transform training data to word vectors
#tokenize each post
nlp = spacy.load("en_core_web_md") #tokenizer
docs = [nlp(nxt) for nxt in X_train]

X_train = [] #empty data set
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


#run training epochs
for epoch in range(num_epochs):
    #training data
    print(f"EPOCH: {epoch+1}/{num_epochs}")
    print("    Posts Trained:")

    #loop through training data
    for i, input in enumerate(X_train):
        optimizer.zero_grad() #zero gradients
        output = model.predict(torch.tensor(np.array(input)), training)[0] #0 is prob that it is snow day, 1 is prob that it isnt

        #criterion requires inputs to be tensors, turn into tesnors
        output_tensor = output.clone().detach().requires_grad_(True)
        expected_tensor = torch.tensor(y_train[i], dtype=torch.float32, requires_grad=True)

        #calculate loss
        loss = criterion(output_tensor, expected_tensor)
        loss.backward() #compute gradient loss
        optimizer.step() #update tensors
        print(f"    {i+1}/{len(X_train)}, loss: {loss}") #training data

#training data
end_time = time.time()
elapsed = end_time - start_time
print(f"Total training time: {elapsed}s")


#enter in sentence to predict
stemmed_phrase = stem("Due to anticipated inclement weather not cancelled but it is an inclement weather day today so dont come to school")
test_doc = nlp(stemmed_phrase) #vectorize phrase
test_x = [token.vector for token in test_doc] #ensure phrase is in tensor form to plug in

#prediction
print(model.predict(torch.tensor(np.array(test_x)), False))


#code written by vincent qu :)
