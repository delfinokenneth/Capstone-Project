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
#-------------------------------------------------------------------------------- FREQUENCY OF WORDS
import collections
import pandas as pd
import matplotlib.pyplot as plt

# Read input file, note the encoding is specified here 
# It may be different in your text file
# file = open('PrideandPrejudice.txt', encoding="utf8")
# a= file.read()
# Stopwords
stopwords = set(line.strip() for line in open('stopwords.txt'))
stopwords = stopwords.union(set(['mr','mrs','one','two','said']))
# Instantiate a dictionary, and for every word in the file, 
# Add to the dictionary if it doesn't exist. If it does, increase the count.
wordcount = {}
# To eliminate duplicates, remember to split by punctuation, and use case demiliters.
#convert the data into list
dataToList = data['comment'].values.tolist()
combineList = ""
for sentence in dataToList:
    combineList += str(sentence)
for word in combineList.lower().split():
    word = word.replace(".","")
    word = word.replace(",","")
    word = word.replace(":","")
    word = word.replace("\"","")
    word = word.replace("!","")
    word = word.replace("â€œ","")
    word = word.replace("â€˜","")
    word = word.replace("*","")
    if word not in stopwords:
        if word not in wordcount:
            wordcount[word] = 1
        else:
            wordcount[word] += 1
# Print most common word
print("The {} most common words are as follows\n".format(10))
word_counter = collections.Counter(wordcount)
print(word_counter)
frequentWords=[]
frequentWordsValues=[]
for word, count in word_counter.most_common(10):
    frequentWords.append(word)
    frequentWordsValues.append(count)
    print(word, ": ", count)
# Close the file
# file.close()
# Create a data frame of the most common words 
# Draw a bar chart
lst = word_counter.most_common(10)
df = pd.DataFrame(lst, columns = ['Word', 'Count'])
df.plot.bar(x='Word',y='Count')

#bar chart for word counter
x = np.array(frequentWords)
y = np.array(frequentWordsValues)

plt.figure(figsize=(6,6))
plt.subplots_adjust(bottom = 0.1)
plt.title("Top 10 Most Common Words in the dataset")
plt.bar(x,y)
plt.show()

#data in labels
count_positive = len(data[ data['label'] == "positive"])
count_negative = len(data[ data['label'] == "negative"])
count_neutral = len(data[ data['label'] == "neutral"])

values = [count_positive,count_negative,count_neutral]
labels = ["positive","negative","neutral"]

list_allColumns = data.values.tolist()
#-------------------------------------------------------------------------------- BAR GRAPH
#bar chart for labels
x = np.array(["positive", "negative", "neutral"])
y = np.array([count_positive, count_negative, count_neutral])

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

#-------------------------------------------------------------------------------- CREATING PIE GRAPH
#display using piegraph
plt.pie(values,labels= labels, autopct='%1.1f%%')
plt.title("Number of Positive, Negative and Neutral")
plt.show()

values = [countCebuano, countEnglish]
labels = ["cebuano","english"]

#plot using piegroph
plt.pie(values,labels= labels, autopct='%1.1f%%')
plt.title("Number of Cebuano and English")
plt.show()

