# -*- coding: utf-8 -*-
"""
Created on Mon May 30 12:36:08 2016
Database: https://archive.ics.uci.edu/ml/datasets/Spambase
@author: Cody Malick, Garrett Smith
"""

import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

def LogisticRegression(training, testing):
    X = []
    y = []    
    
    #Parameters X,Y
    X = training[:,0:56] # Data
    y = training[:,57] # Class Label
    Z = testing[:,0:56] # Testing set
    
    # Function Parameter set
    logistic = linear_model.LogisticRegression()
    logistic.fit(X,y)

    training_results_x = []
    training_results_y = []
    testing_results_x = []
    testing_results_y = []
        
    c = 1.01
    p = 'l2'
    for j in range(0,100):
        c -= .01
        print(logistic.get_params())
        logistic.set_params(**{'C':c,'penalty':p})        
        result = logistic.predict(X)
    
        correct = 0
        incorrect = 0
    
        #print("Training data results:")
        for i in range(0, len(result)):
            if(result[i] == training[i][57]):
                correct += 1
            else:
                incorrect += 1
    
        #print("Correct: " + str(correct) + "| Incorrect: " + str(incorrect))
        training_results_x.append(c)
        training_results_y.append(correct/(correct+incorrect))
    
        logistic.fit(X,y)
        result = logistic.predict(Z)    
    
        correct = 0
        incorrect = 0
        #print("Testing data results:")
        for i in range(0, len(result)):
            if(result[i] == testing[i][57]):
                correct += 1
            else:
                    incorrect += 1
                    
        #print("Correct: " + str(correct) + "| Incorrect: " + str(incorrect))
        testing_results_x.append(c)
        testing_results_y.append(correct/(correct+incorrect))
        
    plt.plot(training_results_x, training_results_y, 'b-')
    plt.plot(testing_results_x, testing_results_y, 'r-')
#        
#    for point in testing_results:
#        plt.plot(point[0], point[1], 'ro')
    plt.xlabel("C Factor")
    plt.ylabel("Accuracy")
    plt.legend(["Training", "Testing"], loc="lower right")
    plt.show()

def main():
    training = np.loadtxt("test_set.csv", delimiter=",")
    testing = np.loadtxt("spambase.data", delimiter=",")
    LogisticRegression(training, testing)


main()