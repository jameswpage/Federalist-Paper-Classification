# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:21:49 2018

In this file, I experiment with different feature sets for classifying the 
federalist papers into authors


@author: James Page
"""



import nltk
import random
import word_counter as wc
import string

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
doc_words = set([item[0] for item in total_word_freq])


def getDocWordSet(doc):
    word_set = set([])
    word_list = doc.split(None)
    for word in word_list:
        word = word.lower()
        word = word.translate(str.maketrans('','', string.punctuation))
        word_set.add(word)
    return word_set
    

#document is a string, not a tuble (doesn't have classification)
def document_features(document):
    document_words = getDocWordSet(document)
    features = {}
    for word in doc_words:
        features['contains({})'.format(word)] = (word in document_words)
    return features




#The following set up code is two create two different sets of data:
#   -train set for training classifier
#   -test set for testing the classifier

#decide what percentage of classified papers will be used to train, remaining
#papers are used to test the classifier performance
train_ratio = 2/3
ham_train_amount = int(len(key_dict['Hamilton']) * train_ratio)
mad_train_amount = int(len(key_dict['Madison']) * train_ratio)
jay_train_amount = int(len(key_dict['Jay']) * train_ratio)

train_set = []
test_set = []
new_test = []
i = j = k = 0
for (d, c) in documents:
    if c == 'Hamilton':
        if i < ham_train_amount:
            train_set.append((document_features(d), c))
        else:
            test_set.append((document_features(d), c))
            new_test.append((d,c))
        i += 1
    elif c == 'Madison':
        if j < mad_train_amount:
            train_set.append((document_features(d), c))
        else:
            test_set.append((document_features(d), c))
            new_test.append((d,c))
        j += 1
    elif c == 'Jay':
        if k < jay_train_amount:
            train_set.append((document_features(d), c))
        else:
            test_set.append((document_features(d), c))
            new_test.append((d,c))
        k += 1
    
    
print(len(train_set), len(test_set))


classifier = nltk.NaiveBayesClassifier.train(train_set)

print(nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(5)
for doc in new_test:
    print(classifier.classify(document_features(doc[0])), doc[1])
