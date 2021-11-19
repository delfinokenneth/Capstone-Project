import numpy as np 
import pandas as pd

import re
import string
import os

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("Comments.csv")
#remove neutral
data = data[data.label != "neutral"]
data['label'] = data["label"].map(lambda x: 1 if x == "positive" else 0)
df_train = data.sample(frac = 0.8)
df_test = data.drop(df_train.index)

print(df_train.head())
print(df_test.head())

def data_cleaning(raw_data):
    raw_data = raw_data.translate(str.maketrans('', '', string.punctuation + string.digits))
    words = raw_data.lower().split()
    stops = set(stopwords.words("english"))
    useful_words = [w for w in words if not w in stops]
    return( " ".join(useful_words))

df_train['comment']=df_train['comment'].apply(data_cleaning)
df_test["comment"]=df_test["comment"].apply(data_cleaning)

import tensorflow as tf
# from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, Flatten
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, SpatialDropout1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers

y = df_train["label"].values
train_reviews = df_train["comment"]
test_reviews = df_test["comment"]

max_features = 6000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_reviews))
list_tokenized_train = tokenizer.texts_to_sequences(train_reviews)
list_tokenized_test = tokenizer.texts_to_sequences(test_reviews)

max_length = 370
X_train = pad_sequences(list_tokenized_train, maxlen=max_length)
X_test = pad_sequences(list_tokenized_test, maxlen=max_length)

def tokenize(sentence):
  max_features = 6000
  tokenizer = Tokenizer(num_words=max_features)
  tokenizer.fit_on_texts(list(train_reviews))
  list_tokenized_test = tokenizer.texts_to_sequences(sentence)
  max_length = 360
  idv_test = pad_sequences(list_tokenized_test, maxlen=max_length)

  return idv_test

def train_model(model, model_name, n_epochs, batch_size, X_data, y_data, validation_split):    

    history = model.fit(
        X_data,
        y_data,
        steps_per_epoch=batch_size,
        epochs=n_epochs,
        validation_split=validation_split,
        verbose=1,
    )
    return history

def generate_graph(history):
    plt.plot(history.history['accuracy'], 'b')
    plt.plot(history.history['val_accuracy'], 'r')
    plt.title('Model Accuracy'),
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


#Random Forest
from sklearn.ensemble import RandomForestClassifier

model_random_forest = RandomForestClassifier(n_estimators = 150, random_state=45, bootstrap = "False", criterion="gini", min_samples_split = 10, min_samples_leaf = 1)
model_random_forest.set_params(max_features=2)
model_random_forest.fit(X_train, y)



def predict_func(model):
  print(X_test)
  prediction = model.predict(X_test)
  print("prediction : ", prediction)
  y_pred = (prediction > 0.5)
  print("y_pred : ", y_pred)

  #df_test["label"] = df_test["label"].map(lambda x: 1 if x == "positive" else 0)
  y_test = df_test["label"]

  cf_matrix = confusion_matrix(y_pred, y_test)
  f1_score_calc = cf_matrix[0][0] / (cf_matrix[0][0] + 0.5 * (cf_matrix[0][1] + cf_matrix[1][0]))
  print('F1-score: %.3f' % f1_score_calc)
  print("Confusion Matrix : ", cf_matrix)

  return f1_score_calc

random_forest_score = predict_func(model_random_forest)

