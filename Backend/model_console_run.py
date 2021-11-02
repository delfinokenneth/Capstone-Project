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

#getting the data (all columns)
allColumns = data[['comment','label','language']]

#getting the data (comment and label columns only)
commentsAndLabel = data[['comment','label']]

#convert comments and label dataFrame into list
list_commentsAndLabel = commentsAndLabel.values.tolist()

#convert all columns dataFrame into list
list_allColumns = allColumns.values.tolist()

#print data
print(type(list_commentsAndLabel))
print(commentsAndLabel.head(10))
print(allColumns.shape)

#Count the data by label
countPositives = sum(p[1] =="positive" for p in list_commentsAndLabel)
countNegatives = sum(p[1] =="negative" for p in list_commentsAndLabel)
countNeutral = sum(p[1] =="neutral" for p in list_commentsAndLabel)

#bar chart for labels
x = np.array(["positive", "negative", "neutral"])
y = np.array([countPositives, countNegatives, countNeutral])

plt.title("Labels in the dataset")
plt.bar(x,y)
plt.show()

#count the data by language
countCebuano = sum(p[2] =="cebuano" for p in list_allColumns)
countEnglish = sum(p[2] =="english" for p in list_allColumns)

#bar chart for languages in the dataset
x = np.array(["cebuano", "english"])
y = np.array([countCebuano, countEnglish])

plt.title("Languages in the dataset")
plt.bar(x,y)
plt.show()

#printing number of data by label
print("No. of positive comments: ", countPositives)
print("No. of negative comments: ", countNegatives)
print("No. of neutral comments: ", countNeutral)

classifier = NaiveBayesClassifier(list_commentsAndLabel)

while(True):
    comment = input("enter you comment here >> ")
    result = classifier.classify(comment)
    print("result: " + result)



