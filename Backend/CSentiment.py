#imports for flask
# import SentimentIntensityAnalyzer class
from flask_mysqldb import MySQL
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

app.config['MYSQL_HOST'] = 'localhost';
app.config['MYSQL_USER'] = 'root';
app.config['MYSQL_PASSWORD'] = '';
app.config['MYSQL_DB'] = 'isent';

mysql = MySQL(app)

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
    vdpos = str(round(sentiment_dict['pos']*100,2))
    vdneu = str(round(sentiment_dict['neu']*100,2))
    vdneg = str(round(sentiment_dict['neg']*100,2))

    print("Sentence Overall Rated As", end = " ") 
    
    #tweak the downpoints of the vader
    #check if "no" exist in the comment
    hasNo = False
    for word in sentence.split():
        if word == "no":
            hasNo = True
            break
        
    if(hasNo
    or "haha" in sentence):
        return NB_Classify(sentence)
    # decide sentiment as positive, negative and neutral 
    elif sentiment_dict['compound'] >= 0.05 : 
        return "positive" + " " + vdpos + " " + vdneu + " " + vdneg
  
    elif sentiment_dict['compound'] <= - 0.05 : 
        return "negative" + " " + vdpos + " " + vdneu + " " + vdneg

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

#clean the dataset, remove words that is in the stopwords
#function for data cleaning
# Stopwords
stopwords = set(line.strip() for line in open('customized_stopwords.txt'))
stopwords = stopwords.union(set(['mr','mrs','one','two','said']))

def data_cleaning(raw_data):
    raw_data = raw_data.translate(str.maketrans('', '', string.punctuation + string.digits))
    words = raw_data.lower().split()
    stops = set(stopwords)
    useful_words = [w for w in words if not w in stops]
    return(" ".join(useful_words))

training['comment']=training['comment'].apply(data_cleaning)

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

    nbpos = str(round(prob.prob("positive"),2))
    nbneu = str(round(prob.prob("neutral"), 2))
    nbneg = str(round(prob.prob("negative"), 2))
    print(comment_blob.classify())
    return comment_blob.classify() + " " + nbpos + " " + nbneu + " " + nbneg

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
    app.run(debug=True)

