
from flask import Flask, redirect, render_template, request, url_for,json,jsonify
from flask_mysqldb import MySQL
from textblob import TextBlob
import numpy as np
import pandas as pd
from textblob.sentiments import NaiveBayesAnalyzer
from googletrans import Translator
from typing import Final
import nltk
from nltk.text import TextCollection
nltk.download('vader_lexicon')
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob.classifiers import NaiveBayesClassifier
import enchant
from werkzeug.exceptions import RequestEntityTooLarge
from array import *
import string


app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost';
app.config['MYSQL_USER'] = 'root';
app.config['MYSQL_PASSWORD'] = '';
app.config['MYSQL_DB'] = 'isent';

mysql = MySQL(app)

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
    
    #tweak the downpoints of the vader
    #check if "no" exist in the comment
    hasNo = False
    for word in sentence.split():
        if word == "no":
            hasNo = True
            break
        
    if(hasNo
    or "haha" in sentence):
        return NB_Classify(sentence)
    # decide sentiment as positive, negative and neutral 
    elif sentiment_dict['compound'] >= 0.05 : 
        return "positive" 
  
    elif sentiment_dict['compound'] <= - 0.05 : 
        return "negative"

    else :
        return NB_Classify(sentence)

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
        return NB_Classify(sentence)

#reading the dataset
data = pd.read_csv('Comments.csv')
print("number of data ", data.shape)
training = data[['comment','label']]

#clean the dataset, remove words that is in the stopwords
#function for data cleaning
# Stopwords
stopwords = set(line.strip() for line in open('customized_stopwords.txt'))
stopwords = stopwords.union(set(['mr','mrs','one','two','said']))

def data_cleaning(raw_data):
    raw_data = raw_data.translate(str.maketrans('', '', string.punctuation + string.digits))
    words = raw_data.lower().split()
    stops = set(stopwords)
    useful_words = [w for w in words if not w in stops]
    return(" ".join(useful_words))

training['comment']=training['comment'].apply(data_cleaning)

#convert comments and label dataFrame into list
list_commentsAndLabel = training.values.tolist()

classifier = NaiveBayesClassifier(list_commentsAndLabel)

def NB_Classify(comment):
    comment_blob = TextBlob(comment, classifier=classifier)

    prob = classifier.prob_classify(comment)
    print("")
    print("positive",round(prob.prob("positive"),2))
    print("negative", round(prob.prob("negative"),2))
    print("neutral",round(prob.prob("neutral"),2))

    return comment_blob.classify()

@app.route("/login.html", methods=["POST","GET"])
def login():
	if request.method == "POST":
		return redirect(url_for("evaluate"))
	else:
		return render_template("login.html")


@app.route("/teachersevaluation", methods=["POST", "GET"])
def evaluate():
	cur = mysql.connection.cursor()
	cur.execute("SELECT * FROM questionaire where section = 1")
	section1 = cur.fetchall()

	cur.execute("SELECT * FROM questionaire where section = 2")
	section2 = cur.fetchall()

	cur.execute("SELECT * FROM questionaire where section = 3")
	section3 = cur.fetchall()

	cur.execute("SELECT * FROM questionaire where section = 4")
	section4 = cur.fetchall()

	cur.execute("SELECT * FROM questionaire where section = 5")
	section5 = cur.fetchall()

	# get comment and sentiment from db
	# changed: satisfied -> positive | unsatisfied -> negative
	cur.execute("SELECT comment,sentiment from evaluation")
	comments = cur.fetchall()

	# get total number of respondents
	cur.execute("select count(id) as totalnum from evaluation")
	numofrespondents = cur.fetchall()

	#get evaluation from sect1
	cur.execute("SELECT SUBSTRING(section1, 1, 1) from evaluation")
	evalsec1 = cur.fetchall()


	# <!-- DB guide-> https://imgur.com/YMKA4ib -->
	cur.execute("""SELECT DISTINCT section.id, section.section, section.name, section.description, section.percentage, 
				(select count(question) from questionaire  where section = '1') as total1, 
				(select count(question) from questionaire  where section = '2') as total2, 
				(select count(question) from questionaire  where section = '3') as total3, 
				(select count(question) from questionaire  where section = '4') as total4,
				(select count(question) from questionaire  where section = '5') as total5 
				from section 
				right join questionaire on section.section = questionaire.section """)
	sectionsleft = cur.fetchall()


	cur.execute(""" SELECT questionaire.section, questionaire.question from questionaire
					right join section
					ON questionaire.section = section.section """)
	sectionsright = cur.fetchall()


	cur.execute("select section1, section2, section3, section4, section5, (select count(id) from evaluation) as totalnum from evaluation")
	evalsecans = cur.fetchall()






	cur.close()

	if request.method == 'POST':
		# Declaring variables for list to store rating in each section
		sec1_rating = []
		sec2_rating = []
		sec3_rating = []
		sec4_rating = []
		sec5_rating = []


		for i in range(len(section1)):
			sec1_rating.append(request.form[f'rating[{i}]'])

		for i in range(len(section2)):
			sec2_rating.append(request.form[f'rating2[{i}]'])

		for i in range(len(section3)):
			sec3_rating.append(request.form[f'rating3[{i}]'])

		for i in range(len(section4)):
			sec4_rating.append(request.form[f'rating4[{i}]'])

		for i in range(len(section5)):
			sec5_rating.append(request.form[f'rating5[{i}]'])



		# code for the translation and getting sentiment analysis
		comment = request.form["txtcomment"]

		cl = NaiveBayesClassifier(training)
		translator = Translator()

		# Lets test the accuracy of the classifier
		# print ("training set accuracy : ", cl.accuracy(train))
		# print ("test set accuracy     : ", cl.accuracy(test))
		if (len(comment) == 0):
			result = "neutral"

		else:
			translated = translator.translate(comment, dest="en")

			# check if the word exist
			count_string = (len(translated.text.strip().split(" ")))

			if (count_string <= 1):
				if (count_string == 0):
					result = "neutral"
				elif (count_string == 1):
					d = enchant.Dict("en_US")
					check_word = d.check(translated.text)

					# counter number of letters (own logic (there is no sentimental word that has
					# 1 or 2 letters only))
					if (len(translated.text) <= 2):
						result = "neutral"
					# if the word exist then classify else neutral
					elif (check_word):
						result = cl.classify(translated.text)
					else:
						result = "neutral"


			else:
				result = cl.classify(translated.text)

		try:
			cur = mysql.connection.cursor()
			# converting list into string
			sec1_string = ','.join(sec1_rating)
			sec2_string = ','.join(sec2_rating)
			sec3_string = ','.join(sec3_rating)
			sec4_string = ','.join(sec4_rating)
			sec5_string = ','.join(sec5_rating)

			sql = "INSERT INTO evaluation (idteacher,idstudent,section1,section2,section3,section4,section5,comment,sentiment)\
			 VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s);"
			val = (
			"18013672", "18013672", sec1_string, sec2_string, sec3_string, sec4_string, sec5_string, comment, result)
			cur.execute(sql, val)
			mysql.connection.commit()
			cur.close()
			return f'<h1>Successfully saved!</h1>'

		except Exception as exp:
			return f'<h1>{exp}</h1>'

	# return redirect(url_for("scratch", sec1_rating = sec1_rating, sec2_rating = sec2_rating, sec3_rating = sec3_rating,
	#			sec4_rating = sec4_rating,sec5_rating = sec5_rating, result = result))


	#		section5 = section5, lensec5 = len(section5),
	#		sectionsleft = sectionsleft,
	#		sectionsright = sectionsright,
	#		lensectionsleft = len(sectionsleft),
	#		lensectionsright = len(sectionsright))

	else:
		return render_template("teachers_evaluation.html",
							   section1=section1, section2=section2,
							   lensec1=len(section1), lensec2=len(section2),
							   section3=section3, lensec3=len(section3),
							   section4=section4, lensec4=len(section4),
							   section5=section5, lensec5=len(section5),
							   datacomments = comments,
							   countrespondents = numofrespondents,
							   evaluationsec1 = evalsec1,
							   lenevalsec1 = len(evalsec1),
							   sectionsleft = sectionsleft,
							   sectionsright = sectionsright,
							   lensectionsleft = len(sectionsleft),
							   lensectionsright = len(sectionsright),
							   evalsecans = evalsecans)




@app.route("/evaluation", methods = ["POST", "GET"])
def evaluation():
	
	cur = mysql.connection.cursor()
	cur.execute("SELECT * FROM questionaire where section = 1")
	section1 = cur.fetchall()

	cur.execute("SELECT * FROM questionaire where section = 2")
	section2 = cur.fetchall()

	cur.execute("SELECT * FROM questionaire where section = 3")
	section3 = cur.fetchall()

	cur.execute("SELECT * FROM questionaire where section = 4")
	section4 = cur.fetchall()

	cur.execute("SELECT * FROM questionaire where section = 5")
	section5= cur.fetchall()


	cur.close()

	if request.method == 'POST':
		#Declaring variables for list to store rating in each section
		sec1_rating	= []
		sec2_rating	= []
		sec3_rating	= []
		sec4_rating	= []
		sec5_rating	= []

		for i in range(len(section1)):
				sec1_rating.append(request.form[f'rating[{i}]'])

		for i in range(len(section2)):
				sec2_rating.append(request.form[f'rating2[{i}]'])

		for i in range(len(section3)):
				sec3_rating.append(request.form[f'rating3[{i}]'])

		for i in range(len(section4)):
				sec4_rating.append(request.form[f'rating4[{i}]'])

		for i in range(len(section5)):
				sec5_rating.append(request.form[f'rating5[{i}]'])

		#code for the translation and getting sentiment analysis
		comment = request.form["txtcomment"]
		comment = comment.replace("miss","")

		result = sentiment_scores(comment)

		try:       
			cur = mysql.connection.cursor()
			#converting list into string
			sec1_string = ','.join(sec1_rating) 
			sec2_string = ','.join(sec2_rating)
			sec3_string = ','.join(sec3_rating) 
			sec4_string = ','.join(sec4_rating) 
			sec5_string = ','.join(sec5_rating)    

			sql = "INSERT INTO evaluation (idteacher,idstudent,section1,section2,section3,section4,section5,comment,sentiment)\
			 VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s);"
			val = ("18013672","18013672",sec1_string,sec2_string,sec3_string,sec4_string,sec5_string,comment,result)
			cur.execute(sql,val)
			mysql.connection.commit()
			cur.close()
			return redirect(url_for("evaluate"))

		except Exception as exp:
			return f'<h1>{exp}</h1>'

		#return redirect(url_for("scratch", sec1_rating = sec1_rating, sec2_rating = sec2_rating, sec3_rating = sec3_rating,
		#			sec4_rating = sec4_rating,sec5_rating = sec5_rating, result = result))

	else:
		return render_template("evaluation_page.html", section1 = section1, section2 = section2, lensec1 = len(section1),
			lensec2= len(section2), section3 = section3, lensec3= len(section3), section4 = section4, lensec4 = len(section4),
			section5 = section5, lensec5 = len(section5))


@app.route("/instrumentevaluation", methods=["POST", "GET"])
def instrument():
	cur = mysql.connection.cursor()
	cur.execute("SELECT * FROM questionaire where section = 1")
	section1 = cur.fetchall()

	cur.execute("SELECT * FROM questionaire where section = 2")
	section2 = cur.fetchall()

	cur.execute("SELECT * FROM questionaire where section = 3")
	section3 = cur.fetchall()

	cur.execute("SELECT * FROM questionaire where section = 4")
	section4 = cur.fetchall()

	cur.execute("SELECT * FROM questionaire where section = 5")
	section5 = cur.fetchall()

	# <!-- DB guide-> https://imgur.com/YMKA4ib -->
	cur.execute("""SELECT DISTINCT section.id, section.section, section.name, section.description, section.percentage, 
				(select count(question) from questionaire  where section = '1') as total1, 
				(select count(question) from questionaire  where section = '2') as total2, 
				(select count(question) from questionaire  where section = '3') as total3, 
				(select count(question) from questionaire  where section = '4') as total4,
				(select count(question) from questionaire  where section = '5') as total5 
				from section 
				right join questionaire on section.section = questionaire.section """)
	sectionsleft = cur.fetchall()


	cur.execute(""" SELECT questionaire.section, questionaire.question from questionaire
					right join section
					ON questionaire.section = section.section """)
	sectionsright = cur.fetchall()

#	cur.execute("SELECT questionaire.section, questionaire.question, section.name, section.description, section.percentage "
#				"FROM questionaire "
#				"RIGHT JOIN section "
#				"ON questionaire.section = section.section")
#			evalsection1

	cur.close()

	if request.method == 'POST':
		# Declaring variables for list to store rating in each section
		sec1_rating = []
		sec2_rating = []
		sec3_rating = []
		sec4_rating = []
		sec5_rating = []

		for i in range(len(section1)):
			sec1_rating.append(request.form[f'rating[{i}]'])

		for i in range(len(section2)):
			sec2_rating.append(request.form[f'rating2[{i}]'])

		for i in range(len(section3)):
			sec3_rating.append(request.form[f'rating3[{i}]'])

		for i in range(len(section4)):
			sec4_rating.append(request.form[f'rating4[{i}]'])

		for i in range(len(section5)):
			sec5_rating.append(request.form[f'rating5[{i}]'])

		# code for the translation and getting sentiment analysis
		comment = request.form["txtcomment"]

		cl = NaiveBayesClassifier(train)
		translator = Translator()

		# Lets test the accuracy of the classifier
		# print ("training set accuracy : ", cl.accuracy(train))
		# print ("test set accuracy     : ", cl.accuracy(test))
		if (len(comment) == 0):
			result = "neutral"

		else:
			translated = translator.translate(comment, dest="en")

			# check if the word exist
			count_string = (len(translated.text.strip().split(" ")))

			if (count_string <= 1):
				if (count_string == 0):
					result = "neutral"
				elif (count_string == 1):
					d = enchant.Dict("en_US")
					check_word = d.check(translated.text)

					# counter number of letters (own logic (there is no sentimental word that has
					# 1 or 2 letters only))
					if (len(translated.text) <= 2):
						result = "neutral"
					# if the word exist then classify else neutral
					elif (check_word):
						result = cl.classify(translated.text)
					else:
						result = "neutral"


			else:
				result = cl.classify(translated.text)

		try:
			cur = mysql.connection.cursor()
			# converting list into string
			sec1_string = ','.join(sec1_rating)
			sec2_string = ','.join(sec2_rating)
			sec3_string = ','.join(sec3_rating)
			sec4_string = ','.join(sec4_rating)
			sec5_string = ','.join(sec5_rating)

			sql = "INSERT INTO evaluation (idteacher,idstudent,section1,section2,section3,section4,section5,comment,sentiment)\
			 VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s);"
			val = (
			"18013672", "18013672", sec1_string, sec2_string, sec3_string, sec4_string, sec5_string, comment, result)
			cur.execute(sql, val)
			mysql.connection.commit()
			cur.close()
			return f'<h1>Successfully saved!</h1>'

		except Exception as exp:
			return f'<h1>{exp}</h1>'

	# return redirect(url_for("scratch", sec1_rating = sec1_rating, sec2_rating = sec2_rating, sec3_rating = sec3_rating,
	#			sec4_rating = sec4_rating,sec5_rating = sec5_rating, result = result))

	else:
		return render_template("instrument_evaluation.html", section1=section1, section2=section2, lensec1=len(section1),
							   lensec2=len(section2), section3=section3, lensec3=len(section3), section4=section4,
							   lensec4=len(section4),
							   section5=section5, lensec5=len(section5),
							   sectionsleft = sectionsleft,
							   sectionsright = sectionsright,
							   lensectionsleft = len(sectionsleft),
							   lensectionsright = len(sectionsright))


@app.route("/<sec1_rating><sec2_rating><sec3_rating><sec4_rating><sec5_rating><result>")
def scratch(sec1_rating, sec2_rating,sec3_rating, sec4_rating,sec5_rating, result):
	return f"<h1>{sec1_rating}{sec2_rating}{sec3_rating}{sec4_rating}{sec5_rating}{result}</h1>"

if __name__ == "__main__":
	app.run(debug =True)