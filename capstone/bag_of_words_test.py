from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import spacy
import en_core_web_md
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import csv



#read csv file and append post descriptions to training data
train_x = []
train_y = []
with open("yrdsb_instagram_posts.csv", encoding="utf8") as csvfile:
    yrdsb_instagram_posts = csv.reader(csvfile)
    for row in yrdsb_instagram_posts:
        if len(row) != 0: #idk why
            train_x.append(row[0])
            train_y.append(row[1])

vectorizer = CountVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)

clf_svm = svm.SVC(kernel = "linear")
clf_svm.fit(train_x_vectors, train_y)

test_x = vectorizer.transform(["Due to anticipated inclement weather cancelled"])
print(clf_svm.predict(test_x))



#word vectors
nlp = spacy.load("en_core_web_md")
docs =  [nlp(text) for text in train_x]
train_x_word_vectors = [x.vector for x in docs]

clf_svm_wv = svm.SVC(kernel = "linear")
clf_svm_wv.fit(train_x_word_vectors, train_y)


stemmer = PorterStemmer()
phrase = "Due to anticipated inclement weather not cancelled but it is an inclement weather day today so dont come to school"

words = word_tokenize(phrase)
stop_words = stopwords.words("english")
stemmed_words = []

for word in words:
    if word not in stop_words: 
        stemmed_words.append(stemmer.stem(word))

stemmed_phrase = " ".join(stemmed_words)


test_x = [stemmed_phrase]
test_docs = [nlp(text) for text in test_x]
test_x_word_vectors = [x.vector for x in test_docs]

print(clf_svm_wv.predict(test_x_word_vectors))

