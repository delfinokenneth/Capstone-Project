# imports for flask
# import SentimentIntensityAnalyzer class

# noinspection PyUnresolvedReferences
from xml.dom.minidom import Document
# noinspection PyUnresolvedReferences
from xml.dom.minidom import Element



import nltk
from flask import Flask, request, json, jsonify
from flask_cors import CORS

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Naive bayes imports
import pandas as pd
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
import string

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
    vdpos = round(sentiment_dict['pos']*100,2)
    vdneu = round(sentiment_dict['neu']*100,2)
    vdneg = round(sentiment_dict['neg']*100,2)
    comval = round(sentiment_dict['compound']*100,2)
    print("compound value: ", str(comval))
    #if neutral value is greater than both positive and negative value, then com us "-"
    if(vdneu > vdpos and vdneu > vdneg):
        vdcom = "-"
    else:
        vdcom = vdpos+-abs(vdneg)
        
        

    print("Sentence Overall Rated As", end = " ") 
    
    #tweak the downpoints of the vader
    #check if "no" exist in the comment
    hasNo = False
    for word in sentence.split():
        if word == "no":
            hasNo = True
            break
        
    if(hasNo
    or "n't" in sentence
    or "haha" in sentence
    or "miss" in sentence
    or "absent" in sentence):
        return NB_Classify(sentence)
    # decide sentiment as positive, negative and neutral 
    elif sentiment_dict['compound'] >= 0.05 : 
        return "positive" + " " + str(vdpos) + " " + str(vdneu) + " " + str(vdneg) + " " + str(vdcom)
  
    elif sentiment_dict['compound'] <= - 0.05 : 
        return "negative" + " " + str(vdpos) + " " + str(vdneu) + " " + str(vdneg) + " " + str(vdcom)

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
    nbpos = round(prob.prob("positive"),2)
    nbneu = round(prob.prob("neutral"), 2)
    nbneg = round(prob.prob("negative"), 2)
    print(comment_blob.classify())

    #if neutral value is greater than both positive and negative value, then com us "-"
    if(nbneu > nbpos and nbneu > nbneg):
        nbcom = "-"
    else:
        nbcom = nbpos+-abs(nbneg)

    return comment_blob.classify() + " " + str(nbpos) + " " + str(nbneu) + " " + str(nbneg) + " " + str(nbcom)

# comment = input("enter comment here: ")
# print(sentiment_scores(comment))

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

@app.route("/getQuestions/<section>", methods = ['GET'])
def getQuestions(section):
    con = mysql.connection.cursor()
    con.execute("Select question from questionaire where section = %s", section)
    questions = con.fetchall()
    return jsonify(questions)

if __name__ == '__main__':
    app.run(host="127.0.0.6", port=8000, debug=True)
#    app.run(debug=True)

