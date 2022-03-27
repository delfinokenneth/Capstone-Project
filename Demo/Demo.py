
import nltk
import pandas as pd
from flask import Flask, make_response, redirect, render_template, request, url_for
from flask_mysqldb import MySQL
from googletrans import Translator
from textblob import TextBlob

#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob.classifiers import NaiveBayesClassifier
import enchant
import string

import pdfkit

path_wkhtmltopdf = 'C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe'
config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
pdfkit.from_url("http://google.com", "out.pdf", configuration=config)

app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost';
app.config['MYSQL_USER'] = 'root';
app.config['MYSQL_PASSWORD'] = '';
app.config['MYSQL_DB'] = 'isent';

mysql = MySQL(app)

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
	# get all comments and sentiments that are not null or empty
	cur.execute("SELECT comment,pos,neu,neg,sentiment,score from evaluation where comment is not null and comment <> ''")
	comments = cur.fetchall()

	#get the average of compound values --- will check this again
	cur.execute("SELECT AVG(score) from evaluation where comment is not null and comment <> '' LIMIT 1")
	comAverage = cur.fetchall()

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
	else:
		return render_template("teachers_evaluation.html",
							   section1=section1, section2=section2,
							   lensec1=len(section1), lensec2=len(section2),
							   section3=section3, lensec3=len(section3),
							   section4=section4, lensec4=len(section4),
							   section5=section5, lensec5=len(section5),
							   datacomments = comments,
							   comAverage = comAverage[0],
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

		#getting the sentiment and details from API
		pos_val = getsentiment(comment).split(" ")[1]
		neu_val = getsentiment(comment).split(" ")[2]
		neg_val = getsentiment(comment).split(" ")[3]
		score_val = getsentiment(comment).split(" ")[4]
		sen_val = getsentiment(comment).split(" ")[0]

		#if sentiment is neutral then score = null
		if sen_val == 'neutral':
			score_val = None

		try:       
			cur = mysql.connection.cursor()
			#converting list into string
			sec1_string = ','.join(sec1_rating) 
			sec2_string = ','.join(sec2_rating)
			sec3_string = ','.join(sec3_rating) 
			sec4_string = ','.join(sec4_rating) 
			sec5_string = ','.join(sec5_rating)    

			#if input comment is not empty
			if comment is not "":
				sql = "INSERT INTO evaluation (idteacher,idstudent,section1,section2,section3,section4,section5,pos,neu,neg,comment,sentiment, score)\
					 VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);"
				val = ("18013672","18013672",sec1_string,sec2_string,sec3_string,sec4_string,sec5_string,pos_val,neu_val,neg_val,comment,sen_val,score_val)
			#else input comment is empty
			else:
				sql = "INSERT INTO evaluation (idteacher,idstudent,section1,section2,section3,section4,section5)\
							 VALUES (%s,%s,%s,%s,%s,%s,%s);"
				val = ("18013672", "18013672", sec1_string, sec2_string, sec3_string, sec4_string, sec5_string)
			cur.execute(sql,val)
			mysql.connection.commit()
			cur.close()
			return redirect(url_for("evaluate"))

		except Exception as exp:
			return f'<h1>{exp}</h1>'

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

	else:
		return render_template("instrument_evaluation.html", section1=section1, section2=section2, lensec1=len(section1),
							   lensec2=len(section2), section3=section3, lensec3=len(section3), section4=section4,
							   lensec4=len(section4),
							   section5=section5, lensec5=len(section5),
							   sectionsleft = sectionsleft,
							   sectionsright = sectionsright,
							   lensectionsleft = len(sectionsleft),
							   lensectionsright = len(sectionsright))
# getting average for positive, negative and neutral
def getPositiveAverage():
		cur = mysql.connection.cursor()
		cur.execute("SELECT AVG(pos) from evaluation where idteacher = 18013672 and score IS NOT NULL  LIMIT 1")
		posAve = cur.fetchall()[0]
		return posAve

def getNegativeAverage():
		cur = mysql.connection.cursor()
		cur.execute("SELECT AVG(neg) from evaluation where idteacher = 18013672 and score IS NOT NULL  LIMIT 1")
		negAve = cur.fetchall()[0]
		return negAve

def getNeutralAverage():
		cur = mysql.connection.cursor()
		cur.execute("SELECT AVG(neu) from evaluation where idteacher = 18013672 and score IS NOT NULL LIMIT 1")
		neuAve = cur.fetchall()[0]
		return neuAve
# end for getting average for positive, negative and neutral
@app.route("/generateReport/<sec1>/<sec2>/<sec3>/<sec4>/<sec5>/<comment>",methods=["POST","GET"])
def generateReport(sec1,sec2,sec3,sec4,sec5,comment):
	try:
		posAve = getPositiveAverage()
		negAve = getNegativeAverage()
		neuAve = getNeutralAverage()
		resp = printReport(sec1,sec2,sec3,sec4,sec5,comment,posAve[0],negAve[0],neuAve[0] )
		return resp
	except:
		return "Can't print report"

#method that will send the input comment to the API and return its response
with app.app_context():
	def getsentiment(comment):
		import requests
		dictToSend = {'comment': comment}
		res = requests.post('http://127.0.0.6:8000/getSentiment', json=dictToSend)
		print('response from server:', res.text)
		dictFromServer = res.json()
		return str(dictFromServer)

with app.app_context():	
	def printReport(sec1,sec2,sec3,sec4,sec5,comment,posAve,negAve,neuAve):
		import requests
		data =[
		("Section1", sec1),
		("Section2", sec2),
		("Section3", sec3),
		("Section4", sec4),
		("Section5", sec5),
		("Comments", comment),
		("Teacher", "Bryan Namoc"), #static value
		("Subject", "MATH 101"), #static value
		("Respondents", "36"), #static value
		("posAve", posAve),
		("negAve", negAve),
		("neuAve", neuAve),
		]
		resp = requests.post('http://127.0.0.6:8000/reportGeneration', json = data, stream=True)
		return resp.raw.read(), resp.status_code, resp.headers.items()

if __name__ == "__main__":
	app.run(host='127.0.0.1', port=8080, debug=True)
	#app.run(debug =True)