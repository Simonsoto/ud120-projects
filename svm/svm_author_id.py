#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#Import required packages
from sklearn import svm
from sklearn.scores import accuracy_score

#Support vector classification
clf = SVC(kernel ="linear")

#Training SVC
t0=time()
clf.train(features_train, labels_train)

print('Training time' + round(time()-t0,3)+ ' s.')



#Accuracy of the classifier
t1 = time()
y_test = clf.predict(features_test)
print("Predciting time:", round(time()-t1, 3), "s")


#accuracy 
accuracy_score(labels_test,y_test)



#########################################################


