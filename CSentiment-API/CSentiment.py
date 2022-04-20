# imports for flask
# import SentimentIntensityAnalyzer class

# noinspection PyUnresolvedReferences
from xml.dom.minidom import Document
# noinspection PyUnresolvedReferences
from xml.dom.minidom import Element
from h11 import Data

#for language detection
from langdetect import detect

import nltk
from flask import Flask, request, json, jsonify, make_response, render_template
from flask_cors import CORS
import pandas as pd
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#for NB
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn.model_selection import train_test_split

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

#this is to allow cross-origin to the backend
cors = CORS(app)

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
    #lowercase the  sentence for uniformity
    sentence = sentence.lower() 
    #words to be remove from the comment
    toRemoveWords=["miss","yes", "idk"]
    #words to remove from vader dict
    toRemoveFromVader = ["weakness","weaknesses","no","natural","serious"]

    for word in toRemoveWords:
        sentence = sentence.replace(word,"")
    
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    for word in toRemoveFromVader:
        sid_obj.lexicon.pop(word)

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

    try:
        langUsed = detect(sentence)
    except Exception as e:
        langUsed = ""
    #detect language used

    # decide sentiment as positive, negative and neutral 
    if sentiment_dict['compound'] >= 0.05 : 
        return "positive" + " " + str(vdpos) + " " + str(vdneu) + " " + str(vdneg) + " " + str(vdscore)
  
    elif sentiment_dict['compound'] <= -0.05 :
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
    if sentiment_dict['compound'] >= 0.05 : 
        return "positive"
  
    elif sentiment_dict['compound'] <= - 0.05 : 
        return "negative"
  
    else :
        return NB_Classify(sentence)

#--------------------------------------------------------------------------------------- NAIVE BAYES


#this is to allow cross-origin to the backend
def preprocess_data(data):
    # Remove package name as it's not relevant
    data = data.drop('language', axis=1)
    
    # Convert text to lowercase
    data['comment'] = data['comment'].str.strip().str.lower()
    return data

def NB_Classify(comment):
    #reading the dataset
    data = pd.read_csv('Comments.csv')
    print("number of data ", data.shape)
    data.head()

    data = preprocess_data(data)

    # Split into training and testing data
    x = data['comment']
    y = data['label']

    x, x_test, y, y_test = train_test_split(x,y, stratify=y, test_size=0.25, random_state=50)

    # Vectorize text reviews to numbers
    vec = CountVectorizer(stop_words='english')
    x = vec.fit_transform(x).toarray()
    x_test = vec.transform(x_test).toarray()

    #load the model
    model = pickle.load(open('NB_Model.pkl', 'rb'))
    result = model.predict_proba((vec.transform([comment])))
    classification = model.predict(vec.transform([comment]))[0]
    print("")
    print("positive",round(result[0][2],2))
    print("negative", round(result[0][0],2))
    print("neutral",round(result[0][1],2))
    nbpos = result[0][2]*100
    print(nbpos)
    nbneu = result[0][1]*100
    print(nbneu)
    nbneg = result[0][0]*100
    print(nbneg)
    print(classification)

    if(isNeutralDefaultVal(nbpos,nbneu,nbneg)):
        nbpos = 0
        nbneu = 100
        nbneg = 0
        classification ="neutral"
    #if neutral value is greater than both positive and negative value, then com us "-"
    #if(nbneu > nbpos and nbneu > nbneg):

    # if nb sentiment is  neutral
    if classification == 'neutral':
        nbscore = '-'
    # if nb sentiment is positive or negative
    else:
        nbscore = nbpos+-abs(nbneg)
        nbscore = nbscore+100
        nbscore = nbscore/2
        nbscore = nbscore/100
        nbscore = abs(nbscore)*5
        nbscore = round (nbscore, 2)

    return classification + " " + str(nbpos) + " " + str(nbneu) + " " + str(nbneg) + " " + str(nbscore)

def isNeutralDefaultVal(pos,neu,neg): 
    neu = round(neu,2)
    pos = round(pos,2)
    neg = round(neg,2)
    defNeu = round(23.385689354275733,2)
    defPos = round(42.233856893542765,2)
    defNeg = round(34.380453752181495,2)
    if (neu == defNeu) and (pos == defPos) and (neg == defNeg):
        return True
# ------------------------------------------------------------------------------------------ END FOR NAIVE BAYES
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

