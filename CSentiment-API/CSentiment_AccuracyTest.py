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

getAccuracy()