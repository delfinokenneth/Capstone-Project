# import SentimentIntensityAnalyzer class 
from typing import Final
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

#Display all predictions, correct guess
def displayPredictions():
    correct = 0
    miss = 0
    numOfNeutralMiss = 0
    list_data = data.values.tolist()
    for value in list_data:
        #print(value)
        sentiment = FinalSentiment(value[0])
        #print("result: ", sentiment , "correct answer: ", value[1])
        if(sentiment == value[1]):
            correct += 1
        else:
            if(sentiment == "neutral"):
                numOfNeutralMiss += 1
            elif("no" in value[0]):
                numOfNeutralMiss += 1
            elif("n't" in value[0]):
                numOfNeutralMiss += 1
            elif("haha" in value[0]):
                numOfNeutralMiss += 1
            elif("miss" in value[0]):
                numOfNeutralMiss += 1
            else:
                miss += 1
                #print("-------------------------------------- MISS")
            
            

    print("correct: ", correct)
    print("neutral miss: ", numOfNeutralMiss)
    print("miss: ", miss)

#get the overall Accuracy of this model
def getAccuracy():
    list_data = data.values.tolist()
    correct = 0
    miss = 0
    for entry in list_data:
        sentiment = FinalSentiment(entry[0])
        if(sentiment == entry[1]):
            correct += 1
        else:
            miss += 1
            
    print("accuracy: ", correct/len(data['comment']))
    print("miss", miss/len(data['comment']))
    print("correct: ", correct)
    print("miss", miss)

#get the pos,neg, neu accuracy
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
        sentiment = FinalSentiment(entry[0])
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
        sentiment = FinalSentiment(entry[0])
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
#Call functions
getAccuracy()
getPosNegNeuAccuracy()
getLanguageAccuracy()

displayPredictions()