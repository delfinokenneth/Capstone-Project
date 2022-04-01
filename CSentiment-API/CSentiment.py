# imports for flask
# import SentimentIntensityAnalyzer class

# noinspection PyUnresolvedReferences
from xml.dom.minidom import Document
# noinspection PyUnresolvedReferences
from xml.dom.minidom import Element
from h11 import Data

import nltk
from flask import Flask, request, json, jsonify, make_response, render_template
from flask_cors import CORS
import pandas as pd
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#replace negate and booster tuples in nltk from csv
vaderconstants = pd.read_csv('vaderconstants.csv')
newnegate = tuple(vaderconstants['negate'])
newbooster = vaderconstants.set_index('booster-key')['booster-value'].to_dict()
nltk.sentiment.vader.VaderConstants.NEGATE = newnegate
nltk.sentiment.vader.VaderConstants.BOOSTER_DICT = newbooster


#Naive bayes imports
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
import string

import pdfkit

path_wkhtmltopdf = 'C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe'
config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
pdfkit.from_url("http://google.com", "out.pdf", configuration=config)

app = Flask(__name__)

#this is to allow cross-origin to the backend
cors = CORS(app)

#this is to modify the SentimentIntensityAnalyzer
new_vader ={
    'strict': -4,
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

#global variables
vdpos = 0
vdneu = 0
vdneg = 2
nbpos = 0
nbneu = 0
nbneg = 0

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
    print(sentiment_dict)
    print("word: ", sentence)
    print("Overall sentiment dictionary is : ", sentiment_dict) 
    print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative")
    print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral") 
    print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive")
    vdpos = sentiment_dict['pos']*100
    print(vdpos)
    vdneu = sentiment_dict['neu']*100
    print(vdneu)
    vdneg = sentiment_dict['neg']*100
    print(vdneg)

    #if vd sentiment is not positive or negative
    if sentiment_dict['compound'] >= 0.05 and sentiment_dict['compound'] <= - 0.05:
        vdscore = '-'
    #if vd sentiment is positive or negative
    else:
        vdscore = vdpos+-abs(vdneg)
        vdscore = vdscore + 100
        vdscore = vdscore/2
        vdscore = vdscore/100
        vdscore = abs(vdscore) * 5
        vdscore = round(vdscore, 2)

    print("Sentence Overall Rated As", end = " ") 
    
    #tweak the downpoints of the vader
    #check if "no" exist in the comment
    hasNo = False
    for word in sentence.split():
        if word == "no":
            hasNo = True
            break
        
    if(hasNo):
        return NB_Classify(sentence)
    # decide sentiment as positive, negative and neutral 
    elif sentiment_dict['compound'] >= 0.05 : 
        return "positive" + " " + str(vdpos) + " " + str(vdneu) + " " + str(vdneg) + " " + str(vdscore)
  
    elif sentiment_dict['compound'] <= -0.05 :
        return "negative" + " " + str(vdpos) + " " + str(vdneu) + " " + str(vdneg) + " " + str(vdscore)

    else :
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
print(data)
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
    nbpos = prob.prob("positive")*100
    print(nbpos)
    nbneu = prob.prob("neutral")*100
    print(nbneu)
    nbneg = prob.prob("negative")*100
    print(nbneg)
    print(comment_blob.classify())

    #if neutral value is greater than both positive and negative value, then com us "-"
    #if(nbneu > nbpos and nbneu > nbneg):

    # if nb sentiment is  neutral
    if comment_blob.classify() == 'neutral':
        nbscore = '-'
    # if nb sentiment is positive or negative
    else:
        nbscore = nbpos+-abs(nbneg)
        nbscore = nbscore+100
        nbscore = nbscore/2
        nbscore = nbscore/100
        nbscore = abs(nbscore)*5
        nbscore = round (nbscore, 2)

    return comment_blob.classify() + " " + str(nbpos) + " " + str(nbneu) + " " + str(nbneg) + " " + str(nbscore)

# comment = input("enter comment here: ")
# print(sentiment_scores(comment))

#convert 2d list into dictionary
def toDict(data):
    labels = [row[0] for row in data]
    values = [row[1] for row in data]
    dataDict = {}
    for i in range(len(values)):
        dataDict[labels[i]] = values[i]

    return dataDict
#creating list for the labels in average chart
def averageChartLabel():
    labels = []
    labels.append("Section 1")
    labels.append("Section 2")
    labels.append("Section 3")
    labels.append("Section 4")
    labels.append("Section 5")
    labels.append("Comment")

    return labels

#creating list for pos,neg,neu averages
def posNegNeuAve(dataDict):
    values = []
    values.append(dataDict['posAve'])
    values.append(dataDict['negAve'])
    values.append(dataDict['neuAve'])

    return values

#creating list for the labels in average chart
def averageChartValues(dataDict):
    values = []
    values.append(dataDict['Section1'])
    values.append(dataDict['Section2'])
    values.append(dataDict['Section3'])
    values.append(dataDict['Section4'])
    values.append(dataDict['Section5'])
    values.append(dataDict['Comments'])

    return values
#building API
@app.route("/getSentiment", methods=['POST'])
def sentimentAnalyis():
    #get the data from the payload
    comment = request.get_json(force=True)
    print("sentiment scores below : ")
    result = sentiment_scores(comment.get("comment"))
    return jsonify(result)

@app.route("/displaydata", methods=['GET'])
def displayData():
    return jsonify(list_commentsAndLabel)
    
#Report Generation
@app.route("/reportGeneration",methods=["POST","GET"])
def generateReport():
    data = request.get_json(force=True)
    dataDict = toDict(data)
    averageLabel = averageChartLabel()
    averageValues = averageChartValues(dataDict)
    sentimentAve = posNegNeuAve(dataDict)
    print("sentiment Averages: ", sentimentAve)
    rendered = render_template("report.html", labels = averageLabel, values=averageValues, data = dataDict, sentimentAve = sentimentAve)
    pdf = pdfkit.from_string(rendered, configuration=config)

    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'inline; filename=summary.pdf'

    return response

if __name__ == '__main__':
    app.run(host="127.0.0.6", port=8000, debug=True)
    #app.run(host="0.0.0.0", port=5000, debug=True)
#    app.run(debug=True)

