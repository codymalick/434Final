# -*- coding: utf-8 -*-
"""
Created on Mon May 30 12:36:08 2016
Database: https://archive.ics.uci.edu/ml/datasets/Spambase
@author: Cody Malick, Garrett Smith
"""

import numpy as np
from sklearn import tree
from sklearn.externals.six import StringIO
import matplotlib.pyplot as plt
from itertools import count 

def DecisionTreeDepth(training, testing, depth):
    X = []
    y = []
    
    #Parameters X,Y
    X = training[:,0:56] # Data
    y = training[:,57] # Class Label
    Z = testing[:,0:56] # Testing set
    if depth == 0:
        dt = tree.DecisionTreeClassifier()
        print("Max tree depth: None")
    else:
        dt = tree.DecisionTreeClassifier(max_depth=depth)
        print("Max tree depth: ", depth)
    print(dt.fit(X,y))
    
    result = dt.predict(X)
    
    correct = 0
    incorrect = 0
    
    print("Training data results:")
    for i in range(0, len(result)):
        if(result[i] == training[i][57]):
            correct += 1
        else:
            incorrect += 1
    
    print("Correct: " + str(correct) + "| Incorrect: " + str(incorrect))
    print("Accuracy: " + str(correct/(correct+incorrect)))    
    
    
    dt.fit(X,y)
    result = dt.predict(Z)    
    
    correct = 0
    incorrect = 0
    print("Testing data results:")
    for i in range(0, len(result)):
        if(result[i] == testing[i][57]):
            correct += 1
        else:
            incorrect += 1
    print("Correct: " + str(correct) + "| Incorrect: " + str(incorrect))
    print("Accuracy: " + str(correct/(correct+incorrect)))   
    return correct/(correct+incorrect)
    
    
def DecisionTreeLeaf(training, testing, leaf_nodes):
    X = []
    y = []
    
    #Parameters X,Y
    X = training[:,0:56] # Data
    y = training[:,57] # Class Label
    Z = testing[:,0:56] # Testing set
    if leaf_nodes == 0:
        dt = tree.DecisionTreeClassifier()
        print("Max tree leaf nodes: None")
    else:
        dt = tree.DecisionTreeClassifier(max_leaf_nodes=leaf_nodes)
        print("Max tree depth: ", leaf_nodes)
    print(dt.fit(X,y))
    
    result = dt.predict(X)
    
    correct = 0
    incorrect = 0
    
    print("Training data results:")
    for i in range(0, len(result)):
        if(result[i] == training[i][57]):
            correct += 1
        else:
            incorrect += 1
    
    print("Correct: " + str(correct) + "| Incorrect: " + str(incorrect))
    print("Accuracy: " + str(correct/(correct+incorrect)))    
    
    
    dt.fit(X,y)
    result = dt.predict(Z)    
    
    correct = 0
    incorrect = 0
    print("Testing data results:")
    for i in range(0, len(result)):
        if(result[i] == testing[i][57]):
            correct += 1
        else:
            incorrect += 1
    print("Correct: " + str(correct) + "| Incorrect: " + str(incorrect))
    print("Accuracy: " + str(correct/(correct+incorrect)))   
    return correct/(correct+incorrect)

def main():
    training = np.loadtxt("test_set.csv", delimiter=",")
    testing = np.loadtxt("spambase.data", delimiter=",")
    results = []
    for i in range(1000):
        accuracy = DecisionTreeDepth(training, testing, i)
        results.append([i, accuracy])
    for point in results:
        plt.plot(point[0], point[1], '*')
    plt.xlabel("Tree Depth")
    plt.ylabel("Accuracy")
    plt.show()
    
    results = []
    accuracy = DecisionTreeLeaf(training, testing, 0)
    results.append([0, accuracy])
    for i in range(2, 1000):
        accuracy = DecisionTreeLeaf(training, testing, i)
        results.append([i, accuracy])
    for point in results:
        plt.plot(point[0], point[1], '*')
    plt.xlabel("Tree Leaf Nodes")
    plt.ylabel("Accuracy")
    plt.show()

        


main()
