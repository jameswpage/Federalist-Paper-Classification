# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:50:23 2018

This is a file to count up the word frequencies of the entire document and return 
the n most common as a dictionary, n <= 1000

@author: James Page
"""


import string


class Counter:
    def __init__(self):
        self.word_freqs = {}
        
    def createFreqs(self):
        
        with open('./Documents/complete.txt', 'r') as fobj:
            doc = fobj.read()
            word_list = doc.split(None)
            for word in word_list:
                word = word.translate(str.maketrans('','', string.punctuation))
                word = word.lower()
                if word not in self.word_freqs:
                    self.word_freqs[word] = 1
                else:
                    self.word_freqs[word] += 1
                    
    def getNlargest(self, N = 500):
        
        self.createFreqs()
        
        dictlist = []
        for key in self.word_freqs:
            temp = [key,self.word_freqs[key]]
            
            
            #18 was chosen as min frequency because ~1000 (1092) words have 
            #a higher frequency
            if temp[1] > 18:
                dictlist.append(temp)
        
        dictlist = sorted(dictlist, key = lambda var: var[1], reverse = True)
        return dictlist[:N]
        

if __name__ == "__main__":
    mycounter = Counter()
    print(len(mycounter.getNlargest(20)))