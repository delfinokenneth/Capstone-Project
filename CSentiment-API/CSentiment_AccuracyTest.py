import pandas as pd
import requests
sampleComment = "She is a good teacher!"

#reading the dataset
data = pd.read_csv('Comments.csv')
print("number of data ", data.shape)
training = data[['comment','label']]
#convert comments and label dataFrame into list

list_commentsAndLabel = training.values.tolist()

def getSentiment(sampleComment):
    dictToSend = {'comment': sampleComment}
    res = requests.post('https://csentiment.herokuapp.com/displaydata', json=dictToSend)
    print('response from server:', res.text)
    dictFromServer = res.json()
    return str(dictFromServer)

def getAccuracy():
    correct = 0
    miss = 0
    for entry in list_commentsAndLabel:
        comment = entry[0];
        print(comment)
        sentiment = getSentiment(comment).split(" ")[0]
        if(sentiment == entry[1]):
            correct += 1
        else:
            print("WRONG - CORRECT: ", entry[1])
            miss += 1
    #accuracy as whole
    print("accuracy: ", correct/len(data['comment']))
    print("miss", miss/len(data['comment']))
    print("correct: ", correct)
    print("miss", miss)

getAccuracy()