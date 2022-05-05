
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
    #else pass to Naive Bayes Model
    else:
        return NB_Classify(sentence)


# ------------------------ NAIVE BAYES

#Pre-process data: remove unnecessary data from the dataset like the "language" column
#data was converted into lowercase for uniformity
def preprocess_data(data):
    #Remove the column language from comment.csv since it will not be used
    #axis = 1 : 1 means 'columns', it is to specify that "language" is a "column"
    data = data.drop('language', axis=1)
    
    # Convert text to lowercase
    try:
        data['comment'] = data['comment'].str.strip().str.lower()
    except Exception as e:
        print(e)

    #return the data after removing the language column and converting it into lowercase     
    return data

#Classifying the comment using the NaiveBayes Model
def NB_Classify(comment):

    #reading the dataset
    data = pd.read_csv('Comments.csv')

    #print the head of the data to check if it was read successfully
    data.head()

    #preprocess the data
    # TO FURTHER UNDERSTAND THIS LINE, CHECK THE preprocess_data FUNCTION
    data = preprocess_data(data)

    # Split into training and testing data
    # x = the data under the comment columns
    x = data['comment']
    # y = the data under the label columns
    y = data['label']

    #this is to split the data 
    # test size was 30% so training size was 70%
    x, x_test, y, y_test = train_test_split(x,y, stratify=y, test_size=0.30, random_state=50)

    # TRAINIG PART >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Vectorize text reviews to numbers (converting text into numbers)
    # stop_words = words that carry a litter meaning so it is ignored or removed
    vec = CountVectorizer(stop_words='english')
    x = vec.fit_transform(x).toarray()
    x_test = vec.transform(x_test).toarray()
    # TRAINING PART <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    #load the model
    # rb = read binary
    model = pickle.load(open('NB_Model.pkl', 'rb'))
    # predict the comment using the model
    # get the proba values: numerical values for pos, neu, and neg
    result = model.predict_proba((vec.transform([comment])))
    #get the classification: "positve", "neutral", "negative"
    classification = model.predict(vec.transform([comment]))[0]

    #get the numerical values for pos,neu,neg the multiply by 100
    #sample: values returned are 0.4887 so multiply 100 to make it 48.87.
    nbpos = result[0][2]*100
    nbneu = result[0][1]*100
    nbneg = result[0][0]*100

    # if nb sentiment is  neutral
    if classification == 'neutral':
        # score is null and represented as '-' instead of returning 2.5 value
        # note: you can ask kenneth or bryan for the explaination in this line
        nbscore = '-'

    # if nb sentiment is positive or negative\
    # this is the formula for score
    else:
        nbscore = nbpos + -abs(nbneg)
        nbscore = nbscore + 100
        nbscore = nbscore / 2
        nbscore = nbscore / 100
        nbscore = abs(nbscore) * 5
        nbscore = round(nbscore, 2)

    #this part is more on pag print sa console nga format
    print("NAIVE BAYES : ", classification, "|", nbscore)
    print("sentence was rated as ", nbpos, "% Positive")
    print("sentence was rated as ", nbneu, "% Neutral")
    print("sentence was rated as ", nbneg, "% Negative")
    print("----------------------------")

    #if returned values are default values of Neutral, pos,neu,neg numerical values and classification should be values below
    # TO UNDERSTAND MORE IN THIS LINE, CHECK isNeutralDefaultVal FUNCTION
    # note: if explaination is need just ask bryan
    if(isNeutralDefaultVal(nbpos,nbneu,nbneg)):
        nbpos = 0
        nbneu = 100
        nbneg = 0
        classification ="neutral"

    #return the values
    return classification + " " + str(nbpos) + " " + str(nbneu) + " " + str(nbneg) + " " + str(nbscore)

#this method is to check if the values return are the default values of Neutral
# note: if explaination is need just ask bryan
def isNeutralDefaultVal(pos,neu,neg): 
    neu = round(neu,2)
    pos = round(pos,2)
    neg = round(neg,2)

    defNeu = round(18.470149253731346,2)
    defPos = round(48.50746268656717,2)
    defNeg = round(33.02238805970149,2)

    #if verified that it is a default value, return True else False
    if (neu == defNeu) and (pos == defPos) and (neg == defNeg):
        return True
    else:
        return False


# ------------------------------------------------------------------------------------------ END FOR NAIVE BAYES
# convert 2d list into dictionary
# used for report
def toDict(data):
    labels = [row[0] for row in data]
    values = [row[1] for row in data]
    dataDict = {}
    for i in range(len(values)):
        dataDict[labels[i]] = values[i]

    return dataDict


# creating list for the labels in average chart
# used for report
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
# used for report
def posNegNeuAve(dataDict):
    values = []
    values.append(dataDict['posAve'])
    values.append(dataDict['negAve'])
    values.append(dataDict['neuAve'])

    return values


# creating list for the labels in average chart
# used for report
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
# this is the api for getSentiment (THIS IS THE ENDPOINT OF OUR ANALYZER)

@app.route("/getSentiment", methods=['POST'])
def sentimentAnalyis():
    # get the data from the payload 
    # data = comment
    comment = request.get_json(force=True)
    # pass the comment to analyzer which is in the "sentiment_scores" method
    result = sentiment_scores(comment.get("comment"))
    # return the jsonify value of the result
    return jsonify(result)

# @app.route("/displaydata", methods=['GET'])
# def displayData():
#     return jsonify(list_commentsAndLabel)

# API/Endpoint for Report Generation
@app.route("/reportGeneration", methods=["POST", "GET"])
def generateReport():
    # get the data from the payload 
    # data = comment
    data = request.get_json(force=True)
    # Convert data into a dictionary
    # TO FURTHER UNDERSTAND THIS LINE, VISIT toDict function
    dataDict = toDict(data)
    # Create list of values for the labels in BAR CHART  in report
    # TO FURTHER UNDERSTAND THIS LINE, VISIT averageChartLabel function
    averageLabel = averageChartLabel()
    # Create list of values for the values in BAR CHART in report
    # TO FURTHER UNDERSTAND THIS LINE, VISIT averageChartValues function
    # The previous line was to get the label, this line is to get the values
    averageValues = averageChartValues(dataDict)
    # Create list of values for the values in PIE CHART in report
    # TO FURTHER UNDERSTAND THIS LINE, VISIT posNegNeuAve function
    sentimentAve = posNegNeuAve(dataDict)

    print("sentiment Averages: ", sentimentAve)

    #pass the values to report.html for the template of the pdf report
    rendered = render_template("report.html", labels=averageLabel, values=averageValues, data=dataDict,
                               sentimentAve=sentimentAve)
    # configure options for the pdf file                                   
    options = {'page-size': 'A4', 'encoding': 'utf-8', 'margin-top': '0.5cm', 'margin-bottom': '0.5cm',
               'margin-left': '0.5cm', 'margin-right': '0.5cm'}
    
    #create a pdf file
    pdf = pdfkit.from_string(rendered, options=options, configuration=config)

    #create response from the pdf file
    response = make_response(pdf)
    
    # set filename and "attachment" is to download the pdf file
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'inline; filename=summary.pdf'

    # return the response
    return response


if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host="127.0.0.6", port=8000, debug=True)
# app.run(host="0.0.0.0", port=5000, debug=True)
#    app.run(debug=True)

