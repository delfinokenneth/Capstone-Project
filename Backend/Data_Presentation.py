#---------------------------------------------------------------------------- IMPORTS
import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import nltk

import tensorflow as tf

#---------------------------------------------------------------------------- READ THE DATA
data = pd.read_csv('Comments.csv')
#get ratio for training data(80%) and test data (20%)
df_train = data.sample(frac = 0.8)
df_test = data.drop(df_train.index)

#check if data was read succesfully
print(df_train.shape)
print(df_test.shape)

print("Shape of the data: ", data.shape)
print(data.head(10))

#print unique values of the Labels and Languages in the data
print("Labels in the data",data["label"].unique())
print("Languages in the data",data["language"].unique())

#-------------------------------------------------------------------------------- DATA VISUALIZATION
#data in labels
count_positive = len(data[ data['label'] == "positive"])
count_negative = len(data[ data['label'] == "negative"])
count_neutral = len(data[ data['label'] == "neutral"])

values = [count_positive,count_negative,count_neutral]
labels = ["positive","negative","neutral"]

#display using piegraph
plt.pie(values,labels= labels, autopct='%1.1f%%')
plt.title("Number of Positive, Negative and Neutral")
plt.show()

#data in language used
count_cebuano = len(data[ data['language'] == 'cebuano'])
count_english = len(data[ data['language'] == 'english'])

values = [count_cebuano, count_english]
labels = ["cebuano","english"]

#plot using piegroph
plt.pie(values,labels= labels, autopct='%1.1f%%')
plt.title("Number of Cebuano and English")
plt.show()

