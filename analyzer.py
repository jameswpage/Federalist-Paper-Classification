# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:21:49 2018

In this file, I experiment with different feature sets for classifying the 
federalist papers into authors


@author: James Page
"""



import nltk
import random
import numpy as np
import word_counter as wc
import string

from nltk.corpus import stopwords


#********************************************************************************************
#*******************CREATE documenents from FED. Papers**************************************
#********************************************************************************************

#create a dict of authors:papers
key_dict = {}
with open('./Documents/key.txt', 'r') as key:
    line = key.readline()
    while line:
        name = line[:line.find(':')]
        docs = line[line.find(':')+1:]
        
        docs = docs.strip().replace(' ', '').split(',')
        key_dict[name] = docs
        line = key.readline()


#use key_dict to create document objects for use in classifiers
documents = []
for name in key_dict:
    if name != 'Unknown' and name != 'Hamilton and Madison':
        for num in key_dict[name]:
            with open('./Documents/Doc_No_'+num+'.txt', 'r') as doc:
                documents.append((doc.read(), name))

#shuffle up the documents
random.shuffle(documents)


#find most common words of entire docuement set
counter = wc.Counter()
total_word_freq = counter.getNlargest()

#remove stop words:
doc_words = set([])
for word in total_word_freq:
    #the first 26 words in the stoplist are pronouns which could be invaluable in 
    #determining the author 
    if word[0] not in set(stopwords.words('english')[26:]):
        doc_words.add(word[0])
        

#********************************************************************************************
#*******************Methods useful in most classifiers***************************************
#********************************************************************************************

#gets the set of words in a docuemnt
def getDocWordSet(doc):
    word_set = set([])
    word_list = doc.split(None)
    for word in word_list:
        word = word.lower()
        word = word.translate(str.maketrans('','', string.punctuation))
        word_set.add(word)
    return word_set

#gets the frequency and the total word count, returned as a dictionary and
#an integer
def getDocWordFreq(doc):
    word_freq = {}
    word_list = doc.split(None)
    for word in word_list:
        word = word.lower()
        word = word.translate(str.maketrans('','', string.punctuation))
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
            
    return word_freq, len(word_list)



#********************************************************************************************
#*******************TRAIN and DEV_TEST sets created******************************************
#********************************************************************************************
    
#The following set up code is two create two different sets of data:
#   -train set for training classifier
#   -test set for testing the classifier

#decide what percentage of classified papers will be used to train, remaining
#papers are used to test the classifier performance
train_ratio = 2/3
ham_train_amount = int(len(key_dict['Hamilton']) * train_ratio)
mad_train_amount = int(len(key_dict['Madison']) * train_ratio)
jay_train_amount = int(len(key_dict['Jay']) * train_ratio)

#these are used by the MB classifier
train_set = []
test_set = []
new_test = []

#this is used by the SVm classifier
SVM_data = {'Train' : {'Docs': [], 'Authors': []}, 'Test' : {'Docs': [], 'Authors': []}}

i = j = k = 0
for (d, c) in documents:
    if c == 'Hamilton':
        if i < ham_train_amount:
            train_set.append((document_features(d), c))
            SVM_data['Train']['Docs'].append(d)
            SVM_data['Train']['Authors'].append(c)
        else:
            test_set.append((document_features(d), c))
            new_test.append((d,c))
            SVM_data['Test']['Docs'].append(d)
            SVM_data['Test']['Authors'].append(c)
        i += 1
    elif c == 'Madison':
        if j < mad_train_amount:
            train_set.append((document_features(d), c))
            SVM_data['Train']['Docs'].append(d)
            SVM_data['Train']['Authors'].append(c)
        else:
            test_set.append((document_features(d), c))
            new_test.append((d,c))
            SVM_data['Test']['Docs'].append(d)
            SVM_data['Test']['Authors'].append(c)
        j += 1
    elif c == 'Jay':
        if k < jay_train_amount:
            train_set.append((document_features(d), c))
            SVM_data['Train']['Docs'].append(d)
            SVM_data['Train']['Authors'].append(c)
        else:
            test_set.append((document_features(d), c))
            new_test.append((d,c))
            SVM_data['Test']['Docs'].append(d)
            SVM_data['Test']['Authors'].append(c)
        k += 1
    
    
    
#********************************************************************************************
#*******************NAIVE BAYES CLASSIFIER***************************************************
#********************************************************************************************
#document is a string, not a tuple (doesn't have classification)
#feature set was chosen using trial and error
#best average so far is a simple contains feature (~80%) though (freq > x%) feature
#has been seen to get up to 87%
def document_features(document):
    document_words = getDocWordSet(document)
    doc_word_freq, total = getDocWordFreq(document)
    features = {}
    for word in doc_words:
        if word in doc_word_freq:
            for i in np.arange(.001, .01, .001):
                features["Freq({}) > {}".format(word, i)] = ((doc_word_freq[word]/total) > i)
        else:
            features["Freq({})".format(word)] = 0 
        features['contains({})'.format(word)] = (word in document_words)
    return features

#this is a main function to run a NB classifier using the above as a feature set
def runNBClassifier():
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(5)
    for doc in new_test:
        print(classifier.classify(document_features(doc[0])), doc[1])

#runNBClassifier()


#********************************************************************************************
#*******************SVM CLASSIFIER***********************************************************
#********************************************************************************************

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

#count_vect = CountVectorizer()
#X_train_counts = count_vect.fit_transform(SVM_data['Train']['Docs'])


#tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

svm_model = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('classifier', SGDClassifier()),])
    
svm_model.fit(SVM_data['Train']['Docs'],SVM_data['Train']['Authors'])

predicted_svm = svm_model.predict(SVM_data['Test']['Docs'])
print(np.mean(predicted_svm == SVM_data['Test']['Authors']))




