from flask import Flask,request, json
import numpy as np
import pandas as pd
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
import matplotlib.pyplot as plt
import csv
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

#this is to allow cross-origin to the backend

#reading the dataset
data = pd.read_csv('Comments.csv')
print("number of data ", data.shape)
training = data[['comment','label']]
#convert comments and label dataFrame into list
list_commentsAndLabel = training.values.tolist()

classifier = NaiveBayesClassifier(list_commentsAndLabel)

#function to get how many items in the dataset will be classified  correctly
def getAccuracy():
    correct = 0
    miss = 0
    for entry in list_commentsAndLabel:
        blob = TextBlob(entry[0], classifier=classifier)
        sentiment = blob.classify()
        if(sentiment == entry[1]):
            correct += 1
        else:
            miss += 1
    print("accuracy: ", correct/len(data['comment']))
    print("miss", miss/len(data['comment']))


getAccuracy()

while(True):
    text = input("Enter: ")
    blob = TextBlob(text, classifier=classifier)
    print (blob.classify())

    prob = classifier.prob_classify(text)
    prob.max()

    print("positive",round(prob.prob("positive"),2))
    print("negative", round(prob.prob("negative"),2))
    print("neutral",round(prob.prob("neutral"),2))


        
