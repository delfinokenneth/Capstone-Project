import pandas as pd
import requests
sampleComment = "She is a good teacher!"

#reading the dataset
data = pd.read_csv('Comments.csv')
print("number of data ", data.shape)
training = data[['comment','label','language']]
#convert comments and label dataFrame into list

list_commentsAndLabel = training.values.tolist()

def getSentiment(sampleComment):
    dictToSend = {'comment': sampleComment}
    res = requests.post('http://127.0.0.6:8000/getSentiment', json=dictToSend)
    
    dictFromServer = res.json()
    return str(dictFromServer)

def getAccuracy():
    correct = 0
    miss = 0
    for entry in list_commentsAndLabel:
        comment = entry[0]
        sentiment = getSentiment(comment).split(" ")[0]
        if(sentiment == entry[1]):
            correct += 1
        else:
            print("WRONG - CORRECT: ", entry[1])
            print(comment)
            print('response from server:', getSentiment(comment))
            miss += 1
    #accuracy as whole
    print("accuracy: ", correct/len(data['comment']))
    print("miss", miss/len(data['comment']))
    print("correct: ", correct)
    print("miss", miss)

def languageAccuracy():
    #Count the number of cebuano and english comments
    countCebuano= sum(p[2] =="cebuano" for p in list_commentsAndLabel)
    countEnglish = sum(p[2] =="english" for p in list_commentsAndLabel)
    print(countCebuano, " ", countEnglish)
    cebuanoCorrect = 0
    cebuanoMiss = 0

    englishCorrect = 0
    englishMiss = 0

    for entry in list_commentsAndLabel:
        comment = entry[0]
        sentiment = getSentiment(comment).split(" ")[0]
        if(sentiment == entry[1]):
            if(entry[2] == "cebuano"):
                cebuanoCorrect += 1
            else:
                englishCorrect += 1
        else:
            # print("WRONG - CORRECT: ", entry[1])
            # print(comment)
            # print('response from server:', getSentiment(comment))
            
            if(entry[2] == "english"):
                cebuanoMiss += 1
            else:
                englishMiss += 1

    #accuracy as whole
    print("accuracy (cebuano): ", cebuanoCorrect/countCebuano)
    print("Correct: ", cebuanoCorrect, ", Miss: ", cebuanoMiss)
    print("accuracy (english): ", englishCorrect/countEnglish)
    print("Correct: ", englishCorrect, ", Miss: ", englishMiss)

def posneunegAccuracy():
    #Count the number of cebuano and english comments
    countNeu = sum(p[1] =="neutral" for p in list_commentsAndLabel)
    countPos = sum(p[1] =="positive" for p in list_commentsAndLabel)
    countNeg = sum(p[1] =="negative" for p in list_commentsAndLabel)
    print(countNeu, " ", countPos, " ", countNeg)

    posCorrect = 0
    posMiss = 0

    neuCorrect = 0
    neuMiss = 0

    negCorrect = 0
    negMiss = 0

    for entry in list_commentsAndLabel:
        comment = entry[0]
        sentiment = getSentiment(comment).split(" ")[0]
        if(sentiment == entry[1]):
            if(entry[1] == "positive"):
                posCorrect += 1
            elif(entry[1] == "neutral"):
                neuCorrect += 1
            else:
                negCorrect += 1
        else:
            # print("WRONG - CORRECT: ", entry[1])
            # print(comment)
            # print('response from server:', getSentiment(comment))
            
            if(entry[1] == "positive"):
                posMiss += 1
            elif(entry[1] == "neutral"):
                neuMiss += 1
            else:
                negMiss += 1

    #accuracy as whole
    print("accuracy (positive): ", posCorrect/countPos)
    print("Correct: ", posCorrect, ", Miss: ", posMiss)
    print("accuracy (neutral): ", neuCorrect/countNeu)
    print("Correct: ", neuCorrect, ", Miss: ", neuMiss)
    print("accuracy (negative): ", negCorrect/countNeg)
    print("Correct: ", negCorrect, ", Miss: ", negMiss)

#getAccuracy()
#languageAccuracy()
posneunegAccuracy()