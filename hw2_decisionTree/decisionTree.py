#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:43:35 2020

@author: dulcemariscal
"""

import csv
import numpy as np
# import matplotlib.pyplot as plt
# import pdb 

import sys

if __name__ == '__main__':
    trainingFile = sys.argv[1]
    testFile= sys.argv[2]
    maxdepth= sys.argv[3]
    train_out= sys.argv[4]
    test_out = sys.argv[5]
    metrics_out= sys.argv[6]
maxdepth=int(maxdepth)


#ge
def openFile(infile):
    with open(infile) as f1:
        reader = csv.reader(f1, delimiter='\t')
        data=[]
        for row in reader:
            data.append(row) #all data in a list


    heather=np.array(data[0])
    data.pop(0)
    data=np.array(data)

    y=data[:,-1]
    f1.close()
    
    return y, data, heather   

#Entropy 
def entropy(data):
    outcomeType, counts =np.unique(data, return_counts=True)
    probs=counts/len(data)
    h_y= - np.sum(probs * np.log2(probs))
    
    return h_y

#Mutual Information 
def mutual_information(X,Y):
    outcomeType, counts =np.unique(X, return_counts=True)
   
    # Specific Entropy
    h_X=[]
    for i in range(len(outcomeType)):
        p=Y[X==outcomeType[i]]
        outcomeType1, counts1 =np.unique(p, return_counts=True)
        probs=counts1/len(p)
        h_X=np.hstack((h_X, - np.sum(probs * np.log2(probs))))
    
    #Conditional Entropy
    probs=counts/len(X)
    Cond_entro= np.sum(probs * h_X)

    #Mutual Information
    I = entropy(Y) - Cond_entro
    return I 


#Pick data of best attribute
def bestAttribute(data, heather):
    y=data[:,-1]
    atributes=data[:,:-1]
    n_atributes=atributes.shape
    m_i=[]
    for c in range(n_atributes[1]):
        m_i =np.hstack((m_i, mutual_information(atributes[:,c],y)))
    index=np.argmax(m_i) 
    
        
    max_mi=m_i[index]    
    b_split=atributes[:,index]
    return b_split , heather[index], index, max_mi
    


def majority_vote(y):
    
    Outcometype, counts=np.unique(y, return_counts=True)
    
    if  len(Outcometype)==1:
        vote=Outcometype[0]
    elif counts[0]> counts[1]:
        vote=Outcometype[0]
    elif counts[0]< counts[1]:
        vote=Outcometype[1]
    elif counts[0]== counts[1]:
        Outcometype.sort()
        vote=Outcometype[-1]

    return vote


def getLabels_errors(y_hat,y):
    if len(y_hat)!=len(y):
        label_clas=[y_hat]*len(y)
        label_clas=np.array(label_clas)
    else:
        label_clas=y_hat

    error=(sum(y!=label_clas))/len(y)

    return  error, label_clas 

#Stump Tree
class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.val = None
        self.depth=None
        self.split=None
        self.attri=None
        self.error=None
        self.h=None
        self.classf=None
        self.data=None
        self.counter=None
      
def stump(data,heather,depth,max_depth,split,label):
     Nodo=Node()   
     
     if len(heather)==0 or len(np.unique(data[:,-1]))==1 or  \
         depth==max_depth or max_depth==0:
        #leaf node
        Decision=majority_vote(data[:,-1]) 
        error, label_clas=  getLabels_errors(Decision,data[:,-1])
        Nodo.error= error
        Nodo.val=Decision
        Nodo.split=split
        Nodo.h=data[:,-1] 
        Nodo.data=data
        Nodo.classf=label_clas
        Nodo.depth=depth+1 
        Nodo.counter=Nodo.depth-1
        
        return Nodo
    
     else:
        Nodo.depth=depth+1 
        Nodo.counter=Nodo.depth-1
        max_depth=max_depth
        b_split, label, index, max_mi= bestAttribute(data,heather)
        
        
        if max_mi<=0:

            Decision=majority_vote(data[:,-1]) 
            error, label_clas= getLabels_errors(Decision,data[:,-1])
            Nodo.error= error
            Nodo.val=Decision
            Nodo.split=split
            Nodo.h=data[:,-1]
            Nodo.data=data
            Nodo.classf=label_clas
            Nodo.depth=depth+1 
            Nodo.counter=Nodo.depth-1
            
            return Nodo 
            
        else: 
            Nodo.attri=label 
            outcomeType =np.unique(b_split)
            Nodo.split=split
            LeftData= data[b_split==outcomeType[1]]
            Nodo.h=data[:,-1] 
            Nodo.data=data
             
            Nodo.left=stump(LeftData, heather,Nodo.depth,max_depth,outcomeType[1],label) 
            
            
            if Nodo.val==None:
                Nodo.attri=label 
                RightData=data[b_split==outcomeType[0]]
                Nodo.split=split
                Nodo.h=data[:,-1]  
                Nodo.data=data
                Nodo.right=stump(RightData, heather,Nodo.depth,max_depth,outcomeType[0],label)

        return Nodo   
     


def printPreorder(root,loop,title,data):    
    labelToCompare=np.unique(data)
    
    if root!=None:
        # First print the data of node
        h_types, times = np.unique(root.h, return_counts=True)
        
        if loop==0:
            print('[',times[0], h_types[0],'/', times[1] ,h_types[1],']')
            loop=+1
        elif loop>0 and len(times)>1:
            print('|'*root.counter,title,'=',root.split,':','[',times[0], h_types[0],'/', times[1] ,h_types[1],']')
            
            loop=+1
        else:
            tt=labelToCompare[labelToCompare!=h_types[0]] 
            if tt[0]==labelToCompare[0]:
                  print('|'*root.counter,title,'=',root.split,':','[',0,tt[0],'/',times[0], h_types[0],']')
                  loop=+1
            else:
                print('|'*root.counter,title,'=',root.split,':','[',times[0], h_types[0], '/', 0,tt[0],']')
                loop=+1
        
        #The recurse on right 
        printPreorder(root.left,loop,root.attri,data)
        
        #Finally recursive on rigth child 
        printPreorder(root.right,loop,root.attri,data)
         
        
        


def predicLabels(root,data,outlabel,heather,a):
    
    if root!=None:
        # a         1=+1
        # print(a)
        if root.attri!=None:
            index=np.where(heather==root.attri)
            response=data[index[0]]
            # print(root.asttri)
            if root.left!=None and root.left.split==response:
                 predicLabels(root.left,data,outlabel,heather,a)
            elif root.right!=None and root.right.split==response: 
                predicLabels(root.right,data,outlabel,heather,a)            
                
        
        elif root.attri==None:
            outlabel.append(root.val)
    return outlabel




def MyTree(File,maxdepth):

    y, data, heather  = openFile(File)
    # pdb.set_trace()
    results= stump(data,heather,0,maxdepth,[],[])
    
    
    printPreorder(results,0,[], results.data[:,-1])

    classLabels=[]

       
    for r in range(data.shape[0]):

        outlabel= predicLabels(results,data[r,:],[],heather,0)    
        classLabels.append(outlabel)

     
    labels=np.array(classLabels)
    labels=labels.flatten()
    error= getLabels_errors(labels,y)[0]
    
    return error, labels, results
    


def test_MyTree(testFile,results_train):
    y, data, heather  = openFile(testFile)
    classLabels=[]
    for r in range(data.shape[0]):
        outlabel= predicLabels(results_train,data[r,:],[],heather,0)    
        classLabels.append(outlabel)

    labels_test=np.array(classLabels)
    labels_test=labels_test.flatten()
    error_test= getLabels_errors(labels_test,y)[0]
    
    return error_test, labels_test



heather  = openFile(trainingFile)[2]
error_t=[]
error_tx=[]

error_train, labels_training, results_train = MyTree(trainingFile,maxdepth)
error_test, label_testing= test_MyTree(testFile,results_train)

outfile = metrics_out 
f2=open(outfile,"w")
text=['error(train): ' ,str(error_train), "\n",'error(test): ', str(error_test)]
f2.writelines(text)
f2.close()


outfile = train_out
f2=open(outfile,"w")
text=labels_training.tolist()
f2.writelines('%s\n' % labelx for labelx in text)
f2.close()


outfile = test_out
f2=open(outfile,"w")
text=label_testing.tolist()
f2.writelines('%s\n' % labelx for labelx in text)
f2.close()



