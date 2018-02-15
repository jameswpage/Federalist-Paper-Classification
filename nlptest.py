# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 18:10:35 2018

@author: James Page
"""







import nltk
nltk.download('names')
import random

#sentence = """At eight o'clock on Thursday morning Arthur didn't feel very good."""
#tokens = nltk.word_tokenize(sentence)

#tagged = nltk.pos_tag(tokens)






def document_features(name):
    return {'last_letter': name[-1], 'last_two' : name[-2:]}
    

featuresets = [(document_features(n), gender) for (n, gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)

print(nltk.classify.accuracy(classifier, test_set))

classifier.show_most_informative_features(5)