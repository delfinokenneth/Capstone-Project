from flask import Flask
import pandas as pd
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

app = Flask(__name__)

data = pd.read_csv('Comments.csv')
data = data[['comment','label','language']]

print(data.head(50))
print(data.shape)

print("Hello world!");

if __name__ == '__main__':
    app.run(debug=True)