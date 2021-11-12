from flask import Flask,request, json
import numpy as np
import pandas as pd
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
import matplotlib.pyplot as plt
import csv
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

# import SentimentIntensityAnalyzer class 
import nltk
nltk.download('vader_lexicon')
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

#this is to allow cross-origin to the backend
cors = CORS(app)


#---------------------------------------------------------------------------- READING DATASET
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
countPositives = sum(p[1] =="postive" for p in list_commentsAndLabel)
countNegatives = sum(p[1] =="negative" for p in list_commentsAndLabel)
countNeutral = sum(p[1] =="neutral" for p in list_commentsAndLabel)

#--------------------------------------------------------------------------- DATA PRESENTATION
values = [countPositives,countNegatives,countNeutral]
labels = ["positive","negative","neutral"]

#display using piegraph
plt.pie(values,labels= labels, autopct='%1.1f%%')
plt.title("Number of Positive, Negative and Neutral")
plt.show()

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

values = [countCebuano, countEnglish]
labels = ["cebuano","english"]

#plot using piegroph
plt.pie(values,labels= labels, autopct='%1.1f%%')
plt.title("Number of Cebuano and English")
plt.show()

#printing number of data by label
print("No. of positive comments: ", countPositives)
print("No. of negative comments: ", countNegatives)
print("No. of neutral comments: ", countNeutral)

#----------------------------------------------------------------------------- SENTIMENT INTENSITY ANALYZER
#this is to modify the SentimentIntensityAnalyzer
new_vader ={
    'strict': -5,
    'absent': -5,
    'high': 1,
    'understands': 2,
    'understand': 2,
    
}

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
    output = ""
    # decide sentiment as positive, negative and neutral 
    if sentiment_dict['compound'] >= 0.05 : 
        print(output, "Positive") 
        output = "Sentence Overall Rated AS Positive"
        return output
  
    elif sentiment_dict['compound'] <= - 0.05 : 
        print(output, "Negative") 
        output = "Sentence Overall Rated AS Negative"
        return output
  
    else :
        print(output, "neutral")
        output = "Sentence Overall Rated AS Neutral"
        return output

    

#function to split the data except of "not"
def connectSomeWords(sentence):
    list_sentence = sentence.split(" ")

    new_list = []
    new_word = ""
    
    i = 0
    while (i < len(list_sentence)):
        if(list_sentence[i] == "not"):
            new_word= list_sentence[i] + " " + list_sentence[i+1]
            new_list.append(new_word) 
            i += 1
        else:
            new_word=list_sentence[i]
            new_list.append(new_word) 
        i += 1
        sentiment_scores(new_word)
        print("")
    return sentiment_scores(sentence)
#----------------------------------------------------------------------------- NAIVEBAYESCLASSIFIER
classifier = NaiveBayesClassifier(list_commentsAndLabel)

#POST Api for receiving the comment then sentiment analysis
@app.route("/sentimentAnalysis", methods=['POST'])
def sentimentAnalyis():
    #get the data from the payload
    comment = json.loads(request.data)
    result = connectSomeWords(comment.get("comment"))
    print(result)
    return result

#POST Api for receiving the comment then sentiment analysis
@app.route("/displayName", methods=['GET'])
def DisplayName():
    return "Bryan Namoc"
    
if __name__ == '__main__':
    app.run(debug=True)