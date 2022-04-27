# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

data = pd.read_csv('Comments.csv')
# Keeping only the neccessary columns
data = data[['comment','label','language']]

data = data[data.label != "neutral"]
data['comment'] = data['comment'].apply(lambda x: x.lower())
data['comment'] = data['comment'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

print(data[ data['label'] == 'positive'].size)
print(data[ data['label'] == 'negative'].size)

for idx,row in data.iterrows():
    row[0] = row[0].replace('rt',' ')
    
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['comment'].values)
X = tokenizer.texts_to_sequences(data['comment'].values)
X = pad_sequences(X)

embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

Y = pd.get_dummies(data['label']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

batch_size = 32
history = model.fit(X_train, Y_train, epochs = 15, batch_size=batch_size, verbose = 1, validation_split=0.2)

def generate_graph(history):
    plt.plot(history.history['accuracy'], 'b')
    plt.plot(history.history['val_accuracy'], 'r')
    plt.title('Model Accuracy'),
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

generate_graph(history)

validation_size = 1500

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
# score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = 2)
# print("score: %.2f" % (score))
# print("acc: %.2f" % (acc))

pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
for x in range(len(X_validate)):
    
    result = model.predict(X_validate[x].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]
   
    if np.argmax(result) == np.argmax(Y_validate[x]):
        if np.argmax(Y_validate[x]) == 0:
            neg_correct += 1
        else:
            pos_correct += 1
       
    if np.argmax(Y_validate[x]) == 0:
        neg_cnt += 1
    else:
        pos_cnt += 1



print("pos_acc", pos_correct/pos_cnt*100, "%")
print("neg_acc", neg_correct/neg_cnt*100, "%")

twt = "bati kaayo siya pagka maestra kay dili mutudlo ug tarong"
#vectorizing the tweet by the pre-fitted tokenizer instance
twt = tokenizer.texts_to_sequences(twt)
#padding the tweet to have exactly the same shape as `embedding_2` input
twt = pad_sequences(twt, maxlen=21, dtype='int32', value=0)
print(twt)
sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
if(np.argmax(sentiment) == 0):
    print("negative")
elif (np.argmax(sentiment) == 1):
    print("positive")

list_data = data.values.tolist()
#function to get how many items in the dataset will be classified  correctly
def getAccuracy():
    correct = 0
    miss = 0
    output = ""
    for entry in list_data:
        #vectorizing the tweet by the pre-fitted tokenizer instance
        twt = tokenizer.texts_to_sequences(entry[0])
        #padding the tweet to have exactly the same shape as `embedding_2` input
        twt = pad_sequences(twt, maxlen=21, dtype='int32', value=0)
        sentiment = model.predict(twt,batch_size=1)[0]
        if(np.argmax(sentiment) == 0):
            output = "negative"
        elif (np.argmax(sentiment) == 1):
            output = "positive"
        if(output == entry[1]):
            correct += 1
        else:
            miss += 1
            
    print("accuracy: ", correct/len(data['comment']))
    print("miss", miss/len(data['comment']))
    print("correct: ", correct)
    print("miss", miss)
#get the cebuano, english prediction accuracy
def getLanguageAccuracy():
    #Count the number of cebuano and english comments
    countCebuano= sum(p[2] =="cebuano" for p in list_data)
    countEnglish = sum(p[2] =="english" for p in list_data)
    
    print("cebuano = ", countCebuano)
    print("english = ", countEnglish)

    ceb_correct = 0
    eng_correct = 0
    miss = 0

    for entry in list_data:
        #vectorizing the tweet by the pre-fitted tokenizer instance
        twt = tokenizer.texts_to_sequences(entry[0])
        #padding the tweet to have exactly the same shape as `embedding_2` input
        twt = pad_sequences(twt, maxlen=21, dtype='int32', value=0)
        sentiment = model.predict(twt,batch_size=1)[0]
        if(np.argmax(sentiment) == 0):
            output = "negative"
        elif (np.argmax(sentiment) == 1):
            output = "positive"
        if(output == entry[1]):
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

#get the pos,neg,ney accuracy
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
          #vectorizing the tweet by the pre-fitted tokenizer instance
        twt = tokenizer.texts_to_sequences(entry[0])
        #padding the tweet to have exactly the same shape as `embedding_2` input
        twt = pad_sequences(twt, maxlen=21, dtype='int32', value=0)
        sentiment = model.predict(twt,batch_size=1)[0]
        if(np.argmax(sentiment) == 0):
            output = "negative"
        elif (np.argmax(sentiment) == 1):
            output = "positive"
        if(output == entry[1]):
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

#getAccuracy()
#getLanguageAccuracy()
#getPosNegNeuAccuracy()