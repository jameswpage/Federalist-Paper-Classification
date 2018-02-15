# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:12:34 2018

@author: James Page
"""



#this file divides up the 
with open("./Documents/complete.txt") as fobj:
    line = fobj.readline()
    while line[:11] != 'FEDERALIST.':
        line = fobj.readline()
    
    
    Ham = []
    Jay = []
    Mad = []
    Unk = []
    MnH = []
    
    num = 1
    new_file = open('./Documents/Doc_No_1.txt', 'w')
    line = fobj.readline()
    while line:
        if line[:11] == 'FEDERALIST.' or line[:11] == 'FEDERALIST ':
            #this is inlcuded because No. 70 is repeated
            if line[15:17] == '70':
                line = fobj.readline()
                while line[:11]!= 'FEDERALIST.' and line[:11] != 'FEDERALIST ':
                    line = fobj.readline()
            new_file.close()
            num += 1
            new_file = open('./Documents/Doc_No_' + str(num) + '.txt', 'w')
        else:
            if line[:11] == 'HAMILTON OR':
                Unk.append(num)
            elif line[:12] == 'HAMILTON AND':
                MnH.append(num)
            elif line[:8] == 'HAMILTON':
                Ham.append(num)
            elif line[:3] == 'JAY':
                Jay.append(num)
            elif line[:7] == 'MADISON':
                Mad.append(num)
            else:   
                new_file.write(line)
        line = fobj.readline()
        
    new_file.close()
    
    key = open('./Documents/key.txt', 'w')
    key.write('Hamilton: ' + str(Ham)[1:len(str(Ham))-1] + '\n')
    key.write('Madison: ' + str(Mad)[1:len(str(Mad))-1] + '\n')
    key.write('Jay: ' + str(Jay)[1:len(str(Jay))-1] + '\n')
    key.write('Unknown: ' + str(Unk)[1: len(str(Unk))-1] + '\n')
    key.write('Hamilton and Madison: ' + str(MnH)[1:len(str(MnH))-1] + '\n')
    key.close()
        
        
        

