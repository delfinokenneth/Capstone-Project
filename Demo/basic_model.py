##train data for the sentiment analysis
train = [
('What an amazing weather.', 'satisfied'),
('this is an amazing idea!', 'satisfied'),
('feel very good about these ideas.', 'satisfied'),
('this is my best performance.', 'satisfied'),
("what an awesome view", 'satisfied'),
('thank you so much', 'satisfied'),
('Thanks','satisfied'),
('Thank','satisfied'),
('I hate this place', 'unsatisfied'),
('am tired of this stuff.', 'slightly unsatisfied'),
('he is my sworn enemy!', 'unsatisfied'),
('my friends is horrible.', 'unsatisfied'),
('she do not know how to', 'unsatisfied'),
('she is beautiful','satisfied'),
('she is ugly', 'unsatisfied'),
('very few give', 'unsatisfied'),
('too big to give','satisfied'),
("rarely enter",'unsatisfied'),
('bati mutudlo','unsatisfied'),
('kamao mutudlo','satisfied'),
('always present','satisfied'),
('always absent','unsatisfied'),
('go ahead and be absent','unsatisfied'),  
('too much to do', 'unsatisfied'),
('gives us time to do', 'satisfied'),
('she comes early', 'satisfied'), 
('mostly comes early', 'slight satisfied'),
('sometimes late', 'slightly unsatisfied'),
('she comes late', 'unsatisfied'),
('she is late sometimes', 'slightly unsatisfied'),
('Ramesh is a friend of mine.', 'slightly satisfied'),
('she is sometimes not in the mood', 'slightly satisfied'),
('gwapa','satisfied'),
('feel and nawng','unsatisfied'),
('he is a good and very kind','satisfied'),
('bad', 'unsatisfied'),
('good','satisfied'),
('little will give a','unsatisfied'),
('size and grade','satisfied'),
('not going on and absent','unsatisfied'),
('not effective','unsatisfied'),
('ineffective', 'unsatisfied'),
('They are not doing their job','unsatisfied'),
('effective', 'satisfied'),
('like','satisfied'),
('I dont like you','unsatisfied'),
('monetary','unsatisfied'),
('no obvious','unsatisfied'),
('clear','satisfied'),
('putang ina','unsatisfied'),
('stupid','unsatisfied'),
('bogo', 'unsatisfied'),
('bright mutudlo','satisfied'),
('fist find a way','satisfied'),
('curiosity finds a way','satisfied'),
('boutan','satisfied'),
('nothing', 'neutral'),
('I don\'t know what to say', 'neutral'),
('I dont know what to say', 'neutral'),
('no comment','neutral'),
('ambot', 'neutral'),
('wa lang','neutral'),
('Don\'t Know Mind','unsatisfied'),
('Knowledge Mutual','satisfied'),
('Slightly grade and grade','unsatisfied'),
('great grade and grade','satisfied'),
('Bati Mutut','unsatisfied'),
('nice to get','satisfied'),
('I like you','satisfied'),
('licks and fingers','satisfied')
 ]
test = [
('the food was great.', 'satisfied'),
("I feel amazing!", 'satisfied'),
('follows schedule','satisfied'),

('I do not want to live anymore', 'unsatisfied'),
('she is not beautiful','unsatisfied'),
("I ain't feeling dandy today.", 'unsatisfied'),

('I have nothing to say', 'neutral')
]

from flask import Flask, redirect, render_template, request, url_for
from flask_mysqldb import MySQL
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from googletrans import Translator
from nltk.text import TextCollection
from textblob.classifiers import NaiveBayesClassifier
import enchant
from array import *



app = Flask(__name__)

print(type(train))
cl = NaiveBayesClassifier(train)
comment = input("Enter comment here >> ")
result = cl.classify(comment.text)

if __name__ == "__main__":
	app.run(debug =True)