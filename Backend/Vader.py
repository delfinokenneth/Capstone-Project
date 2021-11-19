# import SentimentIntensityAnalyzer class 
import nltk
nltk.download('vader_lexicon')
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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
  
    # decide sentiment as positive, negative and neutral 
    if sentiment_dict['compound'] >= 0.05 : 
        print("Positive") 
  
    elif sentiment_dict['compound'] <= - 0.05 : 
        print("Negative") 
  
    else :
        print("neutral")
#function to split the data except of 
def connectSomeWords(list_sentence):
    list_sentence = list_sentence.split(" ")

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
    sentiment_scores(sentence)

sentence = input("Comment here>> ")
connectSomeWords(sentence)

# --------------------------------------------------------------------------- CHECK ACCURACY
import pandas as pd

data = pd.read_csv('Comments.csv')

#getting the data (all columns)
allColumns = data[['comment','label','language']]

#function that will get the final output
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
        return "neutral"

#function to get how many items in the dataset will be classified  correctly(only english)
def getAccuracy():
    correct = 0
    miss = 0
    numOfNeutralMiss = 0
    sample = data[data.language != "cebuano"]
    sample = sample.values.tolist()
    for value in sample:
        print(value)
        sentiment = FinalSentiment(value[0])
        print("result: ", sentiment , "correct answer: ", value[1])
        if(sentiment == value[1]):
            correct += 1
        else:
            
            if(sentiment == "neutral"):
                numOfNeutralMiss += 1
            elif("no" in value[0]):
                numOfNeutralMiss += 1
            elif("won't" in value[0] or "doesn't" in value[0]):
                numOfNeutralMiss += 1
            elif("haha" in value[0]):
                numOfNeutralMiss += 1
            else:
                miss += 1
                print("-------------------------------------- MISS")
            
            

    print("correct: ", correct)
    print("neutral miss: ", numOfNeutralMiss)
    print("miss: ", miss)

getAccuracy()