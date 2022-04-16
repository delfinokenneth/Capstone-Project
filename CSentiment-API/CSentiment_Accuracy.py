#imports for flask
from flask import Flask,request, json, jsonify
import csv
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

# import SentimentIntensityAnalyzer class 
from typing import Final
import nltk
from nltk.text import TextCollection
nltk.download('vader_lexicon')
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#for language detection
from langdetect import detect

#Naive bayes imports
import numpy as np
import pandas as pd
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
import matplotlib.pyplot as plt

#replace negate and booster tuples in nltk from csv
vaderconstants = pd.read_csv('vaderconstants.csv')
newnegate = tuple(vaderconstants['negate'])
newbooster = vaderconstants.set_index('booster-key')['booster-value'].to_dict()
nltk.sentiment.vader.VaderConstants.NEGATE = newnegate
nltk.sentiment.vader.VaderConstants.BOOSTER_DICT = newbooster

#this is to modify the SentimentIntensityAnalyzer
new_vader ={
    'absent': -5,
    'high': 1,
    'understands': 2,
    'understand': 2,
    'late': -4,
    'on time': 2,
    'ontime': 2,
    'on-time': 2,
    'approachable': 4,
    'without': -2,
}

#get cebuano token and sentiment rating from csv
newvaderdata = pd.read_csv('cebuanonewword.csv')
print("number of data ", newvaderdata.shape)
new_vader = newvaderdata.set_index('token')['rating'].to_dict()

#ALGORITHM 1
# function to print sentiments 
# of the sentence. 
def sentiment_scores(sentence): 
    # Create a SentimentIntensityAnalyzer object. 
    sid_obj = SentimentIntensityAnalyzer() 
    sid_obj.lexicon.update(new_vader)

    # polarity_scores method of SentimentIntensityAnalyzer 
    # oject gives a sentiment dictionary. 
    # which contains pos, neg, neu, and compound scores. 
    sentiment_dict = sid_obj.polarity_scores(sentence) 
    print("word: ", sentence)
    print("Overall sentiment dictionary is : ", sentiment_dict) 
    print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative") 
    print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral") 
    print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive") 
  
    print("Sentence Overall Rated As", end = " ") 

    try:
        langUsed = detect(sentence)
    except Exception as e:
        langUsed = ""
    #detect language used

    # decide sentiment as positive, negative and neutral 
    if sentiment_dict['compound'] >= 0.05 : 
        print("VADER: positive")
        return "positive"
  
    elif sentiment_dict['compound'] <= -0.05 :
        print("VADER: negative")
        return "negative"

    elif (langUsed == "tl" or langUsed == "en" or langUsed == "fr"):
        print("VADER: neutral")
        return "neutral"

    else:
        print("NB: ",NB_Classify(sentence))
        return NB_Classify(sentence)

def FinalSentiment(sentence): 
  
    # Create a SentimentIntensityAnalyzer object. 
    sid_obj = SentimentIntensityAnalyzer() 
    sid_obj.lexicon.update(new_vader) 
    sentiment_dict = sid_obj.polarity_scores(sentence) 

    # decide sentiment as positive, negative and neutral 
    if sentiment_dict['compound'] >= 0.05 : 
        return "positive"
  
    elif sentiment_dict['compound'] <= - 0.05 : 
        return "negative"
  
    else :
        return NB_Classify(sentence)

#--------------------------------------------------------------------------------------- NAIVE BAYES


#this is to allow cross-origin to the backend

#reading the dataset
data = pd.read_csv('Comments.csv')
print("number of data ", data.shape)
training = data[['comment','label']]
#convert comments and label dataFrame into list
list_commentsAndLabel = training.values.tolist()

classifier = NaiveBayesClassifier(list_commentsAndLabel)

def NB_Classify(comment):
    comment_blob = TextBlob(comment, classifier=classifier)

    prob = classifier.prob_classify(comment)
    print("")
    print("positive",round(prob.prob("positive"),2))
    print("negative", round(prob.prob("negative"),2))
    print("neutral",round(prob.prob("neutral"),2))

    return comment_blob.classify()

#function to get how many items in the dataset will be classified  correctly
def getAccuracy():
    correct = 0
    miss = 0
    for entry in list_commentsAndLabel:
        comment = entry[0];
        sentiment = sentiment_scores(comment)
        if(sentiment == entry[1]):
            correct += 1
        else:
            miss += 1

    #get how many pos,neg,and neutral in the dataset
    list_data = data.values.tolist()
    #Count the data by label
    countPositives = sum(p[1] =="positive" for p in list_data)
    countNegatives = sum(p[1] =="negative" for p in list_data)
    countNeutral = sum(p[1] =="neutral" for p in list_data)

    #Count the number of cebuano and english comments
    countCebuano= sum(p[2] =="cebuano" for p in list_data)
    countEnglish = sum(p[2] =="english" for p in list_data)
    
    print("cebuano = ", countCebuano)
    print("english = ", countEnglish)

    ceb_correct = 0
    eng_correct = 0
    pos_correct = 0
    neg_correct = 0
    neu_correct = 0
    miss = 0
    lang_miss = 0
    for entry in list_data:
        comment = entry[0];
        sentiment = sentiment_scores(comment)
        if(sentiment == entry[1]):
            if(entry[1] == "positive"):
                pos_correct += 1
            elif(entry[1] == "negative"):
                neg_correct += 1
            else:
                neu_correct += 1
            
            #for checking the language
            if(entry[2] == "cebuano"):
                ceb_correct += 1
            else:
                eng_correct += 1

        else:
            miss += 1
            lang_miss += 1

    #accuracy as whole
    print("accuracy: ", correct/len(data['comment']))
    print("miss", miss/len(data['comment']))
    print("correct: ", correct)
    print("miss", miss)

    #print accuracy of pos,neg,neu guess
    print("positive accuracy: ", pos_correct/countPositives)
    print("negative accuracy: ", neg_correct/countNegatives)
    print("neutral accuracy: ", neu_correct/countNeutral)

    #print the number of correct guess
    print("positive ", countPositives, " correct pos guess ", pos_correct)
    print("negative ", countNegatives, " correct neg guess ", neg_correct)
    print("neutral ", countNeutral, " correct neu guess ", neu_correct)

    #print accuracy of cebuano,english guess
    print("cebuano accuracy: ", ceb_correct/countCebuano)
    print("english accuracy: ", eng_correct/countEnglish)

    #print the number of correct guess
    print("cebuano ", countCebuano, " correct cebuano guess ", ceb_correct)
    print("english ", countEnglish, " correct english guess ", eng_correct)
    print("all correct predictions: ", ceb_correct+eng_correct, ", wrong predictions: ", lang_miss)




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
        comment = entry[0];
        sentiment = sentiment_scores(comment)
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
        comment = entry[0];
        sentiment = sentiment_scores(comment)
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

print("---------------- OUTPUT --------------")
getAccuracy()
#getPosNegNeuAccuracy()
#getLanguageAccuracy()
print("---------------- END OUTPUT --------------")

