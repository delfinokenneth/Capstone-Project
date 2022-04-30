
# IMPORTS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# noinspection PyUnresolvedReferences
from xml.dom.minidom import Document
# noinspection PyUnresolvedReferences
from xml.dom.minidom import Element
from h11 import Data

# for language detection
from langdetect import detect
#for pdf report generation
import pdfkit

import nltk
from flask import Flask, request, json, jsonify, make_response, render_template
from flask_cors import CORS
import pandas as pd

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#this imports are for the NaiveBayes Model
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn.model_selection import train_test_split
#IMPORTS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# note: TO FURTHER UNDERSTAND THIS SECTION PLEASE OPEN THE vaderconstants.csv and make sure to check the column names
# reads the vaderconstants.csv file (naa diri ang custom negate and boosters nga word)
vaderconstants = pd.read_csv('vaderconstants.csv')
#get the negate words from vaderconstants
newnegate = tuple(vaderconstants['negate'])
#get the booster-key(word) and the booster-value from vaderconstants
newbooster = vaderconstants.set_index('booster-key')['booster-value'].to_dict()
# note: "nltk" mao ni gigamit sa pagkuhas scores naa diri ang bag of words, negate and booster
#set nltk.NEGATE with the newnegate values
nltk.sentiment.vader.VaderConstants.NEGATE = newnegate
#set nltk.BOOSTER with the newbooster values
nltk.sentiment.vader.VaderConstants.BOOSTER_DICT = newbooster

# wkhtmltopdf
import pdfkit
import os, sys, subprocess, platform

# note: "wkhtmltopdf" exe file. mao ni gamit para pag print sa pdf file
# if not in deployment (meaning sa local ra gipadagan,  mao ni nga path gamiton para sa wkhtmltopdf)
if platform.system() == "Windows":
    config = pdfkit.configuration(
        wkhtmltopdf=os.environ.get('WKHTMLTOPDF_BINARY', 'C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe'))
# if in deployment nag run (if deployed na mao ni nga path gamiton para sa wkhtmltopdf)
else:
    os.environ['PATH'] += os.pathsep + os.path.dirname(sys.executable)
    WKHTMLTOPDF_CMD = subprocess.Popen(['which', os.environ.get('WKHTMLTOPDF_BINARY', 'wkhtmltopdf')],
                                       stdout=subprocess.PIPE).communicate()[0].strip()
    config = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_CMD)

wk_options = {
    'page-size': 'Letter',
    'orientation': 'landscape',
    # In order to specify command-line options that are simple toggles
    # using this dict format, we give the option the value None
    'no-outline': None,
    'disable-javascript': None,
    'encoding': 'UTF-8',
    'margin-left': '0.1cm',
    'margin-right': '0.1cm',
    'margin-top': '0.1cm',
    'margin-bottom': '0.1cm',
    'lowquality': None,
}
pdfkit.from_url("http://google.com", "out.pdf", configuration=config)

app = Flask(__name__)

# this is to allow cross-origin to the backend
cors = CORS(app)

# note: TO FURTHER UNDERSTAND THIS SECTION OPEN THHE cebuanonewword.csv
# get cebuano words and its weight/values -> vaders
newvaderdata = pd.read_csv('cebuanonewword.csv')
# this is to print the number of rows and columns sa cebuanonewword.csv
print("number of data ", newvaderdata.shape)
# get the token(word) and rating(weight/value) only and store to new_vader nga variable
# new_vader variable will be used to update the nltk.VADER later
new_vader = newvaderdata.set_index('token')['rating'].to_dict()

#check if the language used is cebuano or english
#disregard other language code basta the thought here is to check either english or cebuano ang comment
def isEnglishOrCebuano(langUsed):
    if (langUsed == "tl" or 
        langUsed == "en" or
        langUsed == "fr" or
        langUsed == "ro" or
        langUsed == "so"):
        return True

# ALGORITHM 1
# function to print sentiments 
# of the sentence.
def sentiment_scores(sentence):
    #lowercase the  sentence for uniformity
    sentence = sentence.lower() 
    #words to be remove from the comment because it can cause wrong result
    toRemoveWords=["miss"]
    #words to remove from vader dict
    #some words/vaders are not applicable for teacher's evaluation 
    toRemoveFromVader = ["weakness","weaknesses","no","natural","serious","hahaha","chance","yes","idk"]

    #for each word naa toRemoveWords nga variable iyang i remove sa sentence
    for word in toRemoveWords:
        sentence = sentence.replace(word,"")
    
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    #each word naa sa toRemoveFromVader which is declared above is i remove siya from NLTK.vader nga dictionary
    for word in toRemoveFromVader:
        sid_obj.lexicon.pop(word)

    #new_vader? naa diri ang cebuanonewwords nato nga vader.
    #then this line is to add atong customize vader sa nltk nga vader
    sid_obj.lexicon.update(new_vader)

    # polarity_scores method of SentimentIntensityAnalyzer
    # oject gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(sentence)

    #if compound score is greater than or equal to 0.05 -> sentiment is positive
    if sentiment_dict['compound'] >= 0.05:
        sentiment_output = "positive"
    #if compound score is less than or equal to -0.05 -> sentiment is negative
    elif sentiment_dict['compound'] <= -0.05:
        sentiment_output = "negative"
    #else sentiment is neutral
    else:
        sentiment_output = "neutral"

    #in default, sentiment scores values kay in between 0 -1
    # example ------------------
    # pos = 0.19091231829
    # neg = 0.29091231829
    # neu = 0.39091231829
    # --------------------------
    #so times 100 to make it something like this -> 19.09... which is more preferable to display as percentage
    vdpos = sentiment_dict['pos'] * 100
    vdneu = sentiment_dict['neu'] * 100
    vdneg = sentiment_dict['neg'] * 100

    # if vd sentiment is not positive or negative
    # if vd is neutral, score is null or presented as "-"
    if  not sentiment_dict['compound'] >= 0.05 and not sentiment_dict['compound'] <= -0.05:
        vdscore = '-'

    # if vd sentiment is positive or negative
    # this is the formula for score
    else:
        vdscore = vdpos + -abs(vdneg)
        vdscore = vdscore + 100
        vdscore = vdscore / 2
        vdscore = vdscore / 100
        vdscore = abs(vdscore) * 5
        vdscore = round(vdscore, 2)

    #this part is more on pag print sa console nga format
    print("word: ", sentence)
    print("Overall sentiment dictionary is : ", sentiment_dict)
    print("----------------------------")
    print("VADER : ", sentiment_output, "|", vdscore)
    print("sentence was rated as ", vdpos, "% Positive")
    print("sentence was rated as ", vdneu, "% Neutral")
    print("sentence was rated as ", vdneg, "% Negative")
    print("----------------------------")

    # detect the language used in the sentence
    try:
        langUsed = detect(sentence)
    except Exception as e:
        langUsed = ""

    # decide sentiment as positive, negative and neutral
    #if compound score is greater than or equal to 0.05 -> return positive and its pos,neg,neu, score values
    if sentiment_dict['compound'] >= 0.05:
        return "positive" + " " + str(vdpos) + " " + str(vdneu) + " " + str(vdneg) + " " + str(vdscore)
    #if compound score is less than or equal to -0.05 -> return  negative and its pos,neg,neu, score values
    elif sentiment_dict['compound'] <= -0.05:
        return "negative" + " " + str(vdpos) + " " + str(vdneu) + " " + str(vdneg) + " " + str(vdscore)
    #if the sentence is cebuano or english or sentence is empty return neutral and its pos,neg,neu, score values
    elif (isEnglishOrCebuano(langUsed) or sentence == ""):
        return "neutral" + " " + str(vdpos) + " " + str(vdneu) + " " + str(vdneg) + " " + str(vdscore)

    else:
        #print("pass to NB, langused: ", langUsed)
        return NB_Classify(sentence)


# ------------------------ NAIVE BAYES
#this is to allow cross-origin to the backend
def preprocess_data(data):
    # Remove package name as it's not relevant
    data = data.drop('language', axis=1)
    
    # Convert text to lowercase
    try:
        data['comment'] = data['comment'].str.strip().str.lower()
    except Exception as e:
        print(e)
    return data

def NB_Classify(comment):
    #reading the dataset
    data = pd.read_csv('Comments.csv')
    #print("number of data ", data.shape)
    data.head()

    data = preprocess_data(data)

    # Split into training and testing data
    x = data['comment']
    y = data['label']

    x, x_test, y, y_test = train_test_split(x,y, stratify=y, test_size=0.15, random_state=45)

    # Vectorize text reviews to numbers
    vec = CountVectorizer(stop_words='english')
    x = vec.fit_transform(x).toarray()
    x_test = vec.transform(x_test).toarray()

    #load the model
    model = pickle.load(open('NB_Model.pkl', 'rb'))
    result = model.predict_proba((vec.transform([comment])))
    classification = model.predict(vec.transform([comment]))[0]

    nbpos = result[0][2]*100
    nbneu = result[0][1]*100
    nbneg = result[0][0]*100

    # if nb sentiment is  neutral
    if classification == 'neutral':
        nbscore = '-'
    # if nb sentiment is positive or negative
    else:
        nbscore = nbpos + -abs(nbneg)
        nbscore = nbscore + 100
        nbscore = nbscore / 2
        nbscore = nbscore / 100
        nbscore = abs(nbscore) * 5
        nbscore = round(nbscore, 2)

    print("NAIVE BAYES : ", classification, "|", nbscore)
    print("sentence was rated as ", nbpos, "% Positive")
    print("sentence was rated as ", nbneu, "% Neutral")
    print("sentence was rated as ", nbneg, "% Negative")
    print("----------------------------")

    if(isNeutralDefaultVal(nbpos,nbneu,nbneg)):
        nbpos = 0
        nbneu = 100
        nbneg = 0
        classification ="neutral"
    #if neutral value is greater than both positive and negative value, then com us "-"
    #if(nbneu > nbpos and nbneu > nbneg):



    return classification + " " + str(nbpos) + " " + str(nbneu) + " " + str(nbneg) + " " + str(nbscore)

def isNeutralDefaultVal(pos,neu,neg): 
    neu = round(neu,2)
    pos = round(pos,2)
    neg = round(neg,2)

    defNeu = round(18.4331797235023,2)
    defPos = round(48.540706605222745,2)
    defNeg = round(33.02611367127495,2)

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

# @app.route("/displaydata", methods=['GET'])
# def displayData():
#     return jsonify(list_commentsAndLabel)

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
    #app.run(debug=True)
    app.run(host="127.0.0.6", port=8000, debug=True)
# app.run(host="0.0.0.0", port=5000, debug=True)
#    app.run(debug=True)

