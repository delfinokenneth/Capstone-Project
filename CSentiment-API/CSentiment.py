# imports for flask
# import SentimentIntensityAnalyzer class

# noinspection PyUnresolvedReferences
from xml.dom.minidom import Document
# noinspection PyUnresolvedReferences
from xml.dom.minidom import Element
from h11 import Data

# for language detection
from langdetect import detect

import nltk
from flask import Flask, request, json, jsonify, make_response, render_template
from flask_cors import CORS
import pandas as pd

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# replace negate and booster tuples in nltk from csv
vaderconstants = pd.read_csv('vaderconstants.csv')
newnegate = tuple(vaderconstants['negate'])
newbooster = vaderconstants.set_index('booster-key')['booster-value'].to_dict()
nltk.sentiment.vader.VaderConstants.NEGATE = newnegate
nltk.sentiment.vader.VaderConstants.BOOSTER_DICT = newbooster

# wkhtmltopdf
import pdfkit
import os, sys, subprocess, platform

# if not in deployment
if platform.system() == "Windows":
    config = pdfkit.configuration(
        wkhtmltopdf=os.environ.get('WKHTMLTOPDF_BINARY', 'C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe'))
# if in deployment
else:
    os.environ['PATH'] += os.pathsep + os.path.dirname(sys.executable)
    WKHTMLTOPDF_CMD = subprocess.Popen(['which', os.environ.get('WKHTMLTOPDF_BINARY', 'wkhtmltopdf')],
                                       stdout=subprocess.PIPE).communicate()[0].strip()
    config = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_CMD)

# Naive bayes imports
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
import string

app = Flask(__name__)

# this is to allow cross-origin to the backend
cors = CORS(app)

# get cebuano token and sentiment rating from csv
newvaderdata = pd.read_csv('cebuanonewword.csv')
print("number of data ", newvaderdata.shape)
new_vader = newvaderdata.set_index('token')['rating'].to_dict()


# ALGORITHM 1
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
    if sentiment_dict['compound'] >= 0.05:
        sentiment_output = "positive"
    elif sentiment_dict['compound'] <= -0.05:
        sentiment_output = "negative"
    else:
        sentiment_output = "neutral"

    vdpos = sentiment_dict['pos'] * 100
    vdneu = sentiment_dict['neu'] * 100
    vdneg = sentiment_dict['neg'] * 100
    print("word: ", sentence)
    print("Overall sentiment dictionary is : ", sentiment_dict)
    print("----------------------------")
    print("VADER : ", sentiment_output)
    print("sentence was rated as ", vdpos, "% Positive")
    print("sentence was rated as ", vdneu, "% Neutral")
    print("sentence was rated as ", vdneg, "% Negative")
    print("----------------------------")

    # if vd sentiment is not positive or negative
    if sentiment_dict['compound'] >= 0.05 and sentiment_dict['compound'] <= - 0.05:
        vdscore = '-'
    # if vd sentiment is positive or negative
    else:
        vdscore = vdpos + -abs(vdneg)
        vdscore = vdscore + 100
        vdscore = vdscore / 2
        vdscore = vdscore / 100
        vdscore = abs(vdscore) * 5
        vdscore = round(vdscore, 2)

    try:
        langUsed = detect(sentence)
    except Exception as e:
        langUsed = ""
    # detect language used

    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05:
        return "positive" + " " + str(vdpos) + " " + str(vdneu) + " " + str(vdneg) + " " + str(vdscore)

    elif sentiment_dict['compound'] <= -0.05:
        return "negative" + " " + str(vdpos) + " " + str(vdneu) + " " + str(vdneg) + " " + str(vdscore)

    elif (langUsed == "tl" or langUsed == "en" or langUsed == "fr"):
        return "neutral" + " " + str(vdpos) + " " + str(vdneu) + " " + str(vdneg) + " " + str(vdscore)

    else:
        return NB_Classify(sentence)


def FinalSentiment(sentence):
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()
    sid_obj.lexicon.update(new_vader)
    sentiment_dict = sid_obj.polarity_scores(sentence)

    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05:
        return "positive"

    elif sentiment_dict['compound'] <= - 0.05:
        return "negative"

    else:
        return NB_Classify(sentence)


# ------------------------ NAIVE BAYES


# this is to allow cross-origin to the backend

# reading the dataset
data = pd.read_csv('Comments.csv')
print("number of data ", data.shape)
print(data)
training = data[['comment', 'label']]
# convert comments and label dataFrame into list
list_commentsAndLabel = training.values.tolist()

classifier = NaiveBayesClassifier(list_commentsAndLabel)


def NB_Classify(comment):
    comment_blob = TextBlob(comment, classifier=classifier)

    prob = classifier.prob_classify(comment)
    print("NAIVE BAYES : ", comment_blob.classify())
    nbpos = prob.prob("positive") * 100
    nbneu = prob.prob("neutral") * 100
    nbneg = prob.prob("negative") * 100
    print("sentence was rated as ", nbpos, "% Positive")
    print("sentence was rated as ", nbneu, "% Neutral")
    print("sentence was rated as ", nbneg, "% Negative")

    if (isNeutralDefaultVal(nbpos, nbneu, nbneg)):
        nbpos = 0
        nbneu = 100
        nbneg = 0
    # if neutral value is greater than both positive and negative value, then com us "-"
    # if(nbneu > nbpos and nbneu > nbneg):

    # if nb sentiment is  neutral
    if comment_blob.classify() == 'neutral':
        nbscore = '-'
    # if nb sentiment is positive or negative
    else:
        nbscore = nbpos + -abs(nbneg)
        nbscore = nbscore + 100
        nbscore = nbscore / 2
        nbscore = nbscore / 100
        nbscore = abs(nbscore) * 5
        nbscore = round(nbscore, 2)

    return comment_blob.classify() + " " + str(nbpos) + " " + str(nbneu) + " " + str(nbneg) + " " + str(nbscore)


def isNeutralDefaultVal(pos, neu, neg):
    neu = round(neu, 2)
    pos = round(pos, 2)
    neg = round(neg, 2)
    defNeu = round(47.844433987356894, 2)
    defPos = round(31.34820078980048, 2)
    defNeg = round(20.8073652228426, 2)
    if (neu == defNeu) and (pos == defPos) and (neg == defNeg):
        return True


# ------------------------------------------------------------------------------------------ END FOR NAIVE BAYES
# convert 2d list into dictionary
def toDict(data):
    labels = [row[0] for row in data]
    values = [row[1] for row in data]
    dataDict = {}
    for i in range(len(values)):
        dataDict[labels[i]] = values[i]

    return dataDict


# creating list for the labels in average chart
def averageChartLabel():
    labels = []
    labels.append("Section 1")
    labels.append("Section 2")
    labels.append("Section 3")
    labels.append("Section 4")
    labels.append("Section 5")
    labels.append("Comment")

    return labels


# creating list for pos,neg,neu averages
def posNegNeuAve(dataDict):
    values = []
    values.append(dataDict['posAve'])
    values.append(dataDict['negAve'])
    values.append(dataDict['neuAve'])

    return values


# creating list for the labels in average chart
def averageChartValues(dataDict):
    values = []
    values.append(dataDict['Section1'])
    values.append(dataDict['Section2'])
    values.append(dataDict['Section3'])
    values.append(dataDict['Section4'])
    values.append(dataDict['Section5'])
    values.append(dataDict['Comments'])

    return values


# building API
@app.route("/getSentiment", methods=['POST'])
def sentimentAnalyis():
    # get the data from the payload
    comment = request.get_json(force=True)
    result = sentiment_scores(comment.get("comment"))
    return jsonify(result)


@app.route("/displaydata", methods=['GET'])
def displayData():
    return jsonify(list_commentsAndLabel)


# Report Generation
@app.route("/reportGeneration", methods=["POST", "GET"])
def generateReport():
    data = request.get_json(force=True)
    dataDict = toDict(data)
    averageLabel = averageChartLabel()
    averageValues = averageChartValues(dataDict)
    sentimentAve = posNegNeuAve(dataDict)
    print("sentiment Averages: ", sentimentAve)
    rendered = render_template("report.html", labels=averageLabel, values=averageValues, data=dataDict,
                               sentimentAve=sentimentAve)
    options = {'page-size': 'A4', 'encoding': 'utf-8', 'margin-top': '0.5cm', 'margin-bottom': '0.5cm',
               'margin-left': '0.5cm', 'margin-right': '0.5cm'}
    pdf = pdfkit.from_string(rendered, options=options, configuration=config)

    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=summary.pdf'

    return response


if __name__ == '__main__':
    app.run(debug=True)
#    app.run(host="127.0.0.6", port=8000, debug=True)
# app.run(host="0.0.0.0", port=5000, debug=True)
#    app.run(debug=True)

