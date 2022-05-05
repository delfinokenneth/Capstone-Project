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
blob = TextBlob("She always miss the class", classifier=classifier)
sentiment = blob.classify()
print(sentiment)
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
    print("correct: ", correct)
    print("miss", miss)

#get the pos,neg,ney accuracy
def getPosNegNeuAccuracy():
    #get how many pos,neg,and neutral in the dataset
    list_data = data.values.tolist()
    #Count the data by label
    countPositives = sum(p[1] =="positive" for p in list_data)
    countNegatives = sum(p[1] =="negative" for p in list_data)
    countNeutral = sum(p[1] =="neutral" for p in list_data)

    pos_correct = 0
    neg_correct = 0
    neu_correct = 0
    miss = 0
    for entry in list_data:
        blob = TextBlob(entry[0], classifier=classifier)
        sentiment = blob.classify()
        if(sentiment == entry[1]):
            if(entry[1] == "positive"):
                pos_correct += 1
            elif(entry[1] == "negative"):
                neg_correct += 1
            else:
                neu_correct += 1
        else:
            miss += 1

    #print accuracy of pos,neg,neu guess
    print("positive accuracy: ", pos_correct/countPositives)
    print("negative accuracy: ", neg_correct/countNegatives)
    print("neutral accuracy: ", neu_correct/countNeutral)

    #print the number of correct guess
    print("positive ", countPositives, " correct pos guess ", pos_correct)
    print("negative ", countNegatives, " correct neg guess ", neg_correct)
    print("neutral ", countNeutral, " correct neu guess ", neu_correct)

#get the cebuano, english prediction accuracy
def getLanguageAccuracy():
    list_data = data.values.tolist()
    #Count the number of cebuano and english comments
    countCebuano= sum(p[2] =="cebuano" for p in list_data)
    countEnglish = sum(p[2] =="english" for p in list_data)
    
    print("cebuano = ", countCebuano)
    print("english = ", countEnglish)

    ceb_correct = 0
    eng_correct = 0
    miss = 0

    for entry in list_data:
        blob = TextBlob(entry[0], classifier=classifier)
        sentiment = blob.classify()
        if(sentiment == entry[1]):
            if(entry[2] == "cebuano"):
                ceb_correct += 1
            else:
                eng_correct += 1
        else:
            miss += 1

    #print accuracy of cebuano,english guess
    print("cebuano accuracy: ", ceb_correct/countCebuano)
    print("english accuracy: ", eng_correct/countEnglish)

    #print the number of correct guess
    print("cebuano ", countCebuano, " correct cebuano guess ", ceb_correct)
    print("english ", countEnglish, " correct english guess ", eng_correct)

#call the methods
getAccuracy()
# getPosNegNeuAccuracy()
# getLanguageAccuracy()

# while(True):
#     text = input("Enter: ")
#     blob = TextBlob(text, classifier=classifier)
#     print (blob.classify())

#     prob = classifier.prob_classify(text)
#     prob.max()

#     print("positive",round(prob.prob("positive"),2))
#     print("negative", round(prob.prob("negative"),2))
#     print("neutral",round(prob.prob("neutral"),2))


        
