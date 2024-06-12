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
import random
import pickle
from sklearn.metrics import accuracy_score




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
    if dropout is not None:
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
                Modified input embeddings
        '''

        #apply attention for each head
        head_results = []

        #NOTE: Uses multithreading/cpu parallelization rather than gpu since it's running on a school computer with integrated graphics, potential extra performance isn't worth it for this project training overnight anyways
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.heads) as executor:
            for i in range(self.heads):
                head_results.append(executor.submit(self.attn_block, input, i, training))

            concurrent.futures.wait(head_results)
        
        #add results to input
        for result in head_results:
            input += result.result()

        return input


    def attn_block(self, input, i, training):
        '''
            Singular attention head, computes self scaled dot product attention of entire input matrix with query, key and value matrices i
    
            Args:
                input (FloatTensor): list of word embeddings in input
                i (int): index of query key and value matrix used for attention head
                training (boolean): whether or not the prediction is used for training or not, enables dropout
            Returns:
                modified_input (FloatTensor): modified input embeddings for head
        '''
        delta_value_matrix = torch.zeros(input.shape)

        #loop through each embedding for queries
        for j, E_q in enumerate(input):

            #i assume because multiplying by a vector with requires grad makes it require grad, and we dont want E_q requiring grad or else mult throws error, so need to make copy with no require grad
            with torch.no_grad(): 
                E_q_clone = E_q.clone().detach() #word embedding for query

            query = torch.mv(self.query_tensor[i], E_q_clone) #create query vector
            delta_value = torch.zeros(self.d_embedding//self.heads, requires_grad=True) #delta_value to be added to input embeddings
            scores = [] #attn scores
            values = [] #value_down vectors for each embedding


            #loop through each embedding in context window for keys
            l, r = max(j - self.context_window, 0), min(j + self.context_window, len(input)) #left and right boundary for context window
            for k in range(l, r):
                if k != j: #prevent attention of itself
                    E_k = input[k] #word embedding for key

                    key = torch.mv(self.key_tensor[i], E_k) #mv requires m to come first then v in args
                    values.append(torch.mv(self.value_down_tensor[i], E_k)) #value vector
                    scores.append(attention(query, key, dropout=self.dropout if training else None).unsqueeze(0)) #append score to attention scores


            values = torch.stack(values, dim=0) #convert lists to tensors
            scores = torch.cat(scores) 
            weights = F.softmax(scores, dim=0)

            #add weighted value vectors to delta_value
            for k in range(len(values)):
                delta_value = torch.add(torch.mul(values[k], weights[k]), delta_value)
            delta_value = delta_value.clone().detach().requires_grad_(True) #grad calculation doesnt work otherwise

            #add delta value to input
            delta_value_matrix[j] += torch.mv(self.value_up_tensor[i], delta_value) #.clone().detach() #eats up memory without detaching, i assume since requires grad makes it keep memory

        #returns rather than directly modifies because heads run in parallel
        return delta_value_matrix


        


class feed_forward(nn.Module):
    '''
        Feed forward neural network (fnn) for transformer

        Attributes:
            w_1 (nn.linear): first hidden layer of nn
            w_2 (nn.linear): second hidden layer of nn
            dropout: dropout rate of neurons in fnn

        Methods:
            __init__(self, d_output, d_embedding, dropout=0.1): constructor for object
            forward(self, input, training): forward pass of fnn
    '''

    def __init__(self, d_input, d_output, d_ff, d_embedding, dropout=0.1):
        '''
            Basic constructor, defines all attributes

            Args:
                d_input (int): length of input sentence, currently set to 300
                d_output (int): number of outputs for fnn, currently set to 2
                d_ff (int): dimension of hidden layer                
                d_embedding (int): dimension/number of coordinates in each word embedding
                dropout: dropout rate of neurons in fnn

            Returns:
                None
        '''
        
        super(feed_forward, self).__init__()
        self.w_1 = nn.Linear(d_embedding*d_input, d_ff)
        self.w_2 = nn.Linear(d_ff, d_output)
         
        self.dropout = nn.Dropout(dropout)


    def forward(self, input, training):
        '''
            Forward pass of fnn, performs neural network calculations and classifies results

            Args:
                input (FloatTensor): matrix of word embeddings in input sentence
                training (boolean): whether or not the prediction is used for training or not, enables dropout

            Returns:
                Score output for each word in sentence, index [0] for word i is probability of snow day suggested by word i, [1] is probability of no snow day
        '''

        if not training:
            self.dropout = nn.Dropout(0)

        #apply fnn layers
        input = torch.flatten(input)
        return F.softmax(self.w_2(self.dropout(self.w_1(input).relu())), dim=0)





class transformer(nn.Module):
    '''
        Whole transformer model, combines multiheaded attention, feed forward neural network and positional encoding

        Attributes:
            heads (int): number of attention heads in block
            d_input (int): length of input sentence, currently set to 300
            d_embedding (int): dimension/number of coordinates in each word embedding

        Methods:
            __init__(self, d_output, d_embedding, context_window, heads): constructor for object
            predict(self, input, training): Executes entire model architecture, multiheaded attention, feed forward neural network and positional encoding
    '''

    def __init__(self, d_input, d_output, d_embedding, d_ff, context_window, heads):
        '''
            Basic constructor, defines all attributes

            Args:
                d_input (int): length of input sentence, currently set to 300
                d_output (int): number of outputs for fnn, 1 for binary classification
                d_embedding (int): dimension/number of coordinates in each word embedding
                d_ff (int): dimension of hidden layer in fnn
                context_window (int): size of the attention's context window, i.e. if context_window is x, the window will be x length to the left and right
                heads (int): number of attention heads in block

            Returns:
                None
        '''
        
        super(transformer, self).__init__()
        self.ff = feed_forward(d_input, d_output, d_ff, d_embedding)
        self.attn = multi_head_attn(heads, d_embedding, context_window)
        self.gen_pe = gen_pe
        self.d_embedding = d_embedding
        self.d_input = d_input


    def forward(self, input, training):
        '''
            Executes entire model architecture, multiheaded attention, feed forward neural network and positional encoding

            Args:
                input (FloatTensor): list of word embeddings in input
                training (boolean): whether or not the prediction is used for training or not, enables dropout

            Returns:
                Prediction tensor of fnn, tensor[0] = probability of snowday, tensor[1] = probability of not snowday
        '''

        #modify input with positional encodings
        #input += self.gen_pe(self.d_input, self.d_embedding) 

        #plug sentence into attention then fnn
        input = self.attn(input, training)
        return self.ff(input, training)




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



def shuffle_arrays(a, b):
    '''
    Shuffles two arrays in unison

    Args:
        a (list): first list
        b (list): second list

    Returns:
        Randomly shuffled versions of a and b, shuffled in unison
    '''

    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)

    return a, b




#driver code
if __name__ == "__main__":
    #torch.autograd.set_detect_anomaly(True)
    print("Starting training...")
    start_time = time.time()

    d_input = 300 #max input size
    d_output = 2
    d_embedding = 96
    d_ff = 20
    context_window = 4//2 #context window is on both sides
    heads = 1
    num_epochs = 5
    batch_size = 16

    model = transformer(d_input, d_output, d_embedding, d_ff, context_window, heads) #num outputs, embedding size, num heads
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

    X_train, y_train = shuffle_arrays(X_train, y_train) #shuffle training data in unison


    #transform training data to word vectors
    #tokenize each post
    nlp = spacy.load("en_core_web_sm") #tokenizer
    docs = [nlp(nxt) for nxt in X_train]


    X_train = [] #empty data set
    #for every post
    for i, doc in enumerate(docs):
        X_train.append([])

        #change token to vector representation
        for token in doc:
            X_train[i].append(token.vector)

        X_train[i] = torch.tensor(np.array(X_train[i])) #cast to tensor for pytorch processes


    #cast ints to floats, required for gradient descent
    y_train = list(map(float, y_train))


    #create validation data sets
    X_val = X_train[1500:]
    y_val = y_train[1500:]
    X_train = X_train[:1500]
    y_train = y_train[:1500]


    #training
    #define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)


    #run training epochs
    for epoch in range(num_epochs):
        #training data
        print("-----------------------------------------------------------------------")
        print(f"EPOCH: {epoch+1}/{num_epochs}")
        print("    Batch:")


        #loop through batches
        num_batches = math.ceil(len(X_train) / batch_size)
        start = 0 #training data starting point for each batch
        epoch_start_time = time.time()
        total_accuracy = 0

        for batch_num in range(num_batches):
            print(f"    {batch_num+1}/{int(num_batches)}", end="", flush=True) #training data

            #output and expected values for batch
            output_futures = []
            output_values = []
            expected_values = []

            #loop through training data
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i in range(start, min(start+batch_size, len(X_train))): #zero gradients
                    input = X_train[i]

                    if input.shape[0] < d_input:
                        input = torch.cat((torch.zeros(d_input-input.shape[0], d_embedding), input), dim=0)

                    #model pred
                    output = executor.submit(model, input.clone().detach(), training) #cloned so input doesnt get modified multiple times through epochs

                    #append output and expected to batch output values
                    output_futures.append(output) #unsqueeze for concatenation
                    expected_values.append(torch.tensor(y_train[i], requires_grad=True).unsqueeze(0))

                #wait for all threads to finish
                concurrent.futures.wait(output_futures)
                for future in output_futures:
                    output_values.append(future.result()[0].unsqueeze(0))
                

            #criterion requires args to be tensors, turn into tensors
            output_float = list(map(float, output_values))
            rounded_output = list(map(round, output_float))

            #calculate loss 
            accuracy = accuracy_score(list(map(float, expected_values)), rounded_output)

            optimizer.zero_grad()
            loss = criterion(torch.cat(output_values), torch.cat(expected_values))
            loss.backward() #backward pass
            optimizer.step() #update tensors
            print(f" - loss: {loss}, accuracy: {accuracy}") #training data

            start += batch_size #increment starting pos in training data
            total_accuracy += accuracy


        #validation data:
        val_accuracy = 0
        for i in range(len(X_val)): 
            input = X_val[i]

            if input.shape[0] < d_input:
                input = torch.cat((torch.zeros(d_input-input.shape[0], d_embedding), input), dim=0)

            #model pred
            output = model(input.clone(), False)
            val_accuracy += accuracy_score([round(float(output[0]))], [y_val[i]])
        

        print("-----------------------------------------------------------------------")
        print(f"Epoch {epoch+1} training accuracy: {total_accuracy/num_batches}")
        print(f"Epoch {epoch+1} validation accuracy: {val_accuracy/len(X_val)}")
        print(f"Epoch {epoch+1} training time: {time.time() - epoch_start_time} seconds")
        


    #training data
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total training time: {elapsed} seconds")
    

    #save model to file
    with open("transformerth.pkl", "wb") as file:
        model = pickle.dump(model, file)


    #code written by vincent qu :)
