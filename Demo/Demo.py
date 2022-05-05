from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_mysqldb import MySQL
import platform

app = Flask(__name__)

# if not in deployment
if platform.system() == "Windows":
    # app.config['MYSQL_HOST'] = 'mysql-76692-0.cloudclusters.net';
    # app.config['MYSQL_USER'] = 'dbuser';
    # app.config['MYSQL_PASSWORD'] = 'dbuser123';
    # app.config['MYSQL_DB'] = 'isent';
    # app.config['MYSQL_PORT'] = 14859;
	app.config['MYSQL_HOST'] = 'localhost';
	app.config['MYSQL_USER'] = 'root';
	app.config['MYSQL_PASSWORD'] = '';
	app.config['MYSQL_DB'] = 'isent';
# in deployment
else:
    app.config['MYSQL_HOST'] = 'mysql-76692-0.cloudclusters.net';
    app.config['MYSQL_USER'] = 'dbuser';
    app.config['MYSQL_PASSWORD'] = 'dbuser123';
    app.config['MYSQL_DB'] = 'isent';
    app.config['MYSQL_PORT'] = 14859;

mysql = MySQL(app)


@app.route("/login.html", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        return redirect(url_for("evaluate"))
    else:
        return render_template("login.html")


# functions related to /teachersevaluation
def isDefaultUrlForSummary(teacher, subject):
    if teacher == "all" and subject == "all":
        return True
    else:
        return False


# get teacher name from teacher id
def getSelectedTeacherName(teacher):
    if (teacher == "all" or teacher == "0"):
        return "All"
    else:
        cur = mysql.connection.cursor()
        sql = "SELECT fname,lname from teachers WHERE idteacher = %s LIMIT 1"
        val = (teacher,)
        cur.execute(sql, val)
        resultTeachers = cur.fetchall()
        return resultTeachers[0][1] + ", " + resultTeachers[0][0]


# get teacher name from subject id
def getSelectedSubjectTitle(teacher, subject):
    cur = mysql.connection.cursor()
    if (subject == "all" or subject == "0"):
        return "All"
    else:
        if (subject == "0"):
            sql = "SELECT title from subjects WHERE teacherId = %s ORDER BY title asc LIMIT 1"
            val = (teacher,)
        else:
            sql = "SELECT edpcode,title from subjects WHERE edpCode = %s LIMIT 1"
            val = (subject,)
        cur.execute(sql, val)
        resultSubject = cur.fetchall()
        return str(resultSubject[0][0]) + "-" + resultSubject[0][1]


# get number of respondents from filtered teacher and subject
def getNumberOfRespondents(teacher, subject):
    cur = mysql.connection.cursor()
    # if default
    # if no teacher is selected, and no subject select (no filter)
    if ((teacher == "all" or teacher == "0") and (subject == "0" or subject == "all")):
        cur.execute("SELECT count(*) from evaluation")
        result = cur.fetchall()
        return result[0][0]
    # if not default
    else:
        # if a teacher is selected, but no subject is selected (teacher + all subjects)
        if ((teacher != "0" and teacher != "all") and (subject == "0" or subject == "all")):
            sql = "SELECT count(*) from evaluation WHERE idTeacher = %s"
            val = (teacher,)
        # if a teacher is selected, and a subject is selected (teacher + subject)
        elif ((teacher != "0" and teacher != "all") and (subject != "0" or subject != "all")):
            sql = "SELECT count(*) from evaluation WHERE idTeacher = %s and edpCode = %s"
            val = (teacher, subject,)
        # else (if there is no teacher selected and a subject is selected)
        else:
            sql = "SELECT count(*) from evaluation WHERE edpCode = %s"
            val = (subject,)

        cur.execute(sql, val)
        result = cur.fetchall()
        return result[0][0]


# get comment,pos,neg,neu
def getSentimentValues(teacher, subject):
    cur = mysql.connection.cursor()
    # if default
    # if no teacher is selected, and no subject select (no filter)
    if ((teacher == "all" or teacher == "0") and (subject == "0" or subject == "all")):
        cur.execute(
            "SELECT comment,pos,neu,neg,sentiment,score from evaluation where comment is not null and comment <> ''")
        return cur.fetchall()
    # if not default
    else:
        # if a teacher is selected, but no subject is selected (teacher + all subjects)
        if ((teacher != "0" and teacher != "all") and (subject == "0" or subject == "all")):
            sql = "SELECT comment,pos,neu,neg,sentiment,score from evaluation where comment is not null and comment <> '' and idteacher = %s"
            val = (teacher,)
        # if a teacher is selected, and a subject is selected (teacher + subject)
        elif ((teacher != "0" and teacher != "all") and (subject != "0" or subject != "all")):
            sql = "SELECT comment,pos,neu,neg,sentiment,score from evaluation where comment is not null and comment <> '' and idteacher = %s and edpCode = %s"
            val = (teacher, subject,)
        # else (if there is no teacher selected and a subject is selected)
        else:
            sql = "SELECT comment,pos,neu,neg,sentiment,score from evaluation where comment is not null and comment <> '' and edpCode = %s"
            val = (subject,)

        cur.execute(sql, val)
        return cur.fetchall()


# get all section rating records
def getRatingValues(teacher, subject):
    cur = mysql.connection.cursor()
    # if default
    # if no teacher is selected, and no subject select (no filter)
    if ((teacher == "all" or teacher == "0") and (subject == "0" or subject == "all")):
        cur.execute(
            "select section1, section2, section3, section4, section5, (select count(id) from evaluation) as totalnum from evaluation")
        return cur.fetchall()
    # if not default
    else:
        # if a teacher is selected, but no subject is selected (teacher + all subjects)
        if ((teacher != "0" and teacher != "all") and (subject == "0" or subject == "all")):
            sql = "select section1, section2, section3, section4, section5, (select count(id) from evaluation WHERE idteacher = %s) as totalnum from evaluation WHERE idteacher = %s"
            val = (teacher, teacher,)
        # if a teacher is selected, and a subject is selected (teacher + subject)
        elif ((teacher != "0" and teacher != "all") and (subject != "0" or subject != "all")):
            sql = "select section1, section2, section3, section4, section5, (select count(id) from evaluation WHERE idteacher = %s and edpCode = %s) as totalnum from evaluation WHERE idteacher = %s and edpCode = %s"
            val = (teacher, subject, teacher, subject,)
        # else (if there is no teacher selected and a subject is selected)
        else:
            sql = "select section1, section2, section3, section4, section5, (select count(id) from evaluation WHERE edpCode = %s) as totalnum from evaluation WHERE edpCode = %s"
            val = (subject, subject,)

        cur.execute(sql, val)
        return cur.fetchall()


@app.route("/teachersevaluation/<teacher>/<subject>", methods=["POST", "GET"])
def evaluate(teacher, subject):
    cur = mysql.connection.cursor()
    global G_TEACHER_NAME
    global G_SUBJECT_NAME
    global G_NUMBER_OF_RESPONDENTS
    global G_TEACHER_ID
    global G_SUBJECT_ID

    G_TEACHER_ID = teacher
    G_SUBJECT_ID = subject
    G_TEACHER_NAME = getSelectedTeacherName(teacher)
    G_SUBJECT_NAME = getSelectedSubjectTitle(teacher, subject)
    G_NUMBER_OF_RESPONDENTS = getNumberOfRespondents(teacher, subject)

    # common queries
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
    # End for common queries

    # FOR DEFAULT QUERIES
    # queries for the teachers and subject filter
    sql = "SELECT * FROM teachers order by (case idteacher when %s then 0 else 1 end), lname asc"
    val = (teacher,)
    cur.execute(sql, val)
    teachers = cur.fetchall()

    if (subject != "all" and (teacher != "0")):
        sql = "SELECT * FROM subjects WHERE teacherId = %s or edpCode = 0 order by (case edpCode when %s then 0 else 1 end), title asc"
        val = (teachers[0][0], subject,)
        cur.execute(sql, val)
        subjects = cur.fetchall()
    else:
        sql = "SELECT * FROM subjects order by (case edpCode when %s then 0 else 1 end), title asc"
        val = (subject,)
        cur.execute(sql, val)
        subjects = cur.fetchall()
    # end for default queries

    # get sentiment values
    comments = getSentimentValues(teacher, subject)

    # get total number of respondents
    numofrespondents = getNumberOfRespondents(teacher, subject)

    # get rating records from all sections
    evalsecans = getRatingValues(teacher, subject)

    # END FOR FILTER RECORDS
    cur.close()

    return render_template("teachers_evaluation.html",
                           section1=section1, section2=section2,
                           lensec1=len(section1), lensec2=len(section2),
                           section3=section3, lensec3=len(section3),
                           section4=section4, lensec4=len(section4),
                           section5=section5, lensec5=len(section5),
                           datacomments=comments,
                           countrespondents=numofrespondents,
                           sectionsleft=sectionsleft,
                           sectionsright=sectionsright,
                           lensectionsleft=len(sectionsleft),
                           lensectionsright=len(sectionsright),
                           evalsecans=evalsecans,
                           teachers=teachers,
                           subjects=subjects,
                           isDefault=isDefaultUrlForSummary(teacher, subject)
                           )


@app.route("/evaluation/<teacher>/<subject>", methods=["POST", "GET"])
def evaluation(teacher, subject):
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

    # FOR DEFAULT QUERIES
    # queries for the teachers and subject filter
    sql = "SELECT * FROM teachers order by (case idteacher when %s then 0 else 1 end), lname asc"
    val = (teacher,)
    cur.execute(sql, val)
    teachers = cur.fetchall()

    sql = "SELECT * FROM subjects WHERE teacherId = %s order by (case edpCode when %s then 0 else 1 end), title asc"
    val = (teachers[0][0], subject,)
    cur.execute(sql, val)
    subjects = cur.fetchall()
    # end for default queries

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
        comment = comment.replace("miss", "")

        # getting the sentiment and details from API
        pos_val = getsentiment(comment).split(" ")[1]
        neu_val = getsentiment(comment).split(" ")[2]
        neg_val = getsentiment(comment).split(" ")[3]
        score_val = getsentiment(comment).split(" ")[4]
        sen_val = getsentiment(comment).split(" ")[0]
        # if sentiment is neutral then score = null
        if sen_val == 'neutral':
            score_val = None

        try:
            cur = mysql.connection.cursor()
            # converting list into string
            sec1_string = ','.join(sec1_rating)
            sec2_string = ','.join(sec2_rating)
            sec3_string = ','.join(sec3_rating)
            sec4_string = ','.join(sec4_rating)
            sec5_string = ','.join(sec5_rating)

            # if input comment is not empty
            if comment is not "":
                # if subject is 0 or default, get the value of first subject
                if (subject == "0" and subject != "all"):
                    subject = subjects[0][0]

                sql = "INSERT INTO evaluation (idteacher,idstudent,edpCode,section1,section2,section3,section4,section5,pos,neu,neg,comment,sentiment, score)\
					 VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);"
                val = (
                teacher, "18013672", subject, sec1_string, sec2_string, sec3_string, sec4_string, sec5_string, pos_val,
                neu_val, neg_val, comment, sen_val, score_val)
            # else input comment is empty
            else:
                sql = "INSERT INTO evaluation (idteacher,idstudent,edpCode,section1,section2,section3,section4,section5)\
							 VALUES (%s,%s,%s,%s,%s,%s,%s,%s);"
                val = (teacher, "18013672", subject, sec1_string, sec2_string, sec3_string, sec4_string, sec5_string)
            cur.execute(sql, val)
            mysql.connection.commit()
            cur.close()
            return redirect("/teachersevaluation/all/all")

        except Exception as exp:
            return f'<h1>{exp}</h1>'

    else:
        return render_template("evaluation_page.html",
                               section1=section1, section2=section2,
                               lensec1=len(section1), lensec2=len(section2),
                               section3=section3, lensec3=len(section3),
                               section4=section4, lensec4=len(section4),
                               section5=section5, lensec5=len(section5),
                               teachers=teachers, subjects=subjects)


# getting average for positive, negative and neutral
def getPositiveAverage():
    teacher = G_TEACHER_ID
    subject = G_SUBJECT_ID

    cur = mysql.connection.cursor()
    # if default
    # if no teacher is selected, and no subject select (no filter)
    if ((teacher == "all" or teacher == "0") and (subject == "0" or subject == "all")):
        cur.execute("SELECT AVG(pos) from evaluation WHERE score IS NOT NULL ")
        posAve = cur.fetchall()[0]
        return posAve
    # if not default
    else:
        # if a teacher is selected, but no subject is selected (teacher + all subjects)
        if ((teacher != "0" and teacher != "all") and (subject == "0" or subject == "all")):
            sql = "SELECT AVG(pos) from evaluation where idteacher = %s and score IS NOT NULL "
            val = (teacher,)
        # if a teacher is selected, and a subject is selected (teacher + subject)
        elif ((teacher != "0" and teacher != "all") and (subject != "0" or subject != "all")):
            sql = "SELECT AVG(pos) from evaluation where idteacher = %s and edpCode = %s and score IS NOT NULL "
            val = (teacher, subject,)
        # else (if there is no teacher selected and a subject is selected)
        else:
            sql = "SELECT AVG(pos) from evaluation where edpCode = %s and score IS NOT NULL "
            val = (subject,)

        cur.execute(sql, val)
        posAve = cur.fetchall()[0]
        return posAve


def getNegativeAverage():
    teacher = G_TEACHER_ID
    subject = G_SUBJECT_ID

    cur = mysql.connection.cursor()
    # if default
    # if no teacher is selected, and no subject select (no filter)
    if ((teacher == "all" or teacher == "0") and (subject == "0" or subject == "all")):
        cur.execute("SELECT AVG(neg) from evaluation WHERE score IS NOT NULL ")
        negAve = cur.fetchall()[0]
        return negAve
    # if not default
    else:
        # if a teacher is selected, but no subject is selected (teacher + all subjects)
        if ((teacher != "0" and teacher != "all") and (subject == "0" or subject == "all")):
            sql = "SELECT AVG(neg) from evaluation where idteacher = %s and score IS NOT NULL"
            val = (teacher,)
        # if a teacher is selected, and a subject is selected (teacher + subject)
        elif ((teacher != "0" and teacher != "all") and (subject != "0" or subject != "all")):
            sql = "SELECT AVG(neg) from evaluation where idteacher = %s and edpCode = %s and score IS NOT NULL"
            val = (teacher, subject,)
        # else (if there is no teacher selected and a subject is selected)
        else:
            sql = "SELECT AVG(neg) from evaluation where edpCode = %s and score IS NOT NULL"
            val = (subject,)

        cur.execute(sql, val)
        negAve = cur.fetchall()[0]
        return negAve


def getNeutralAverage():
    teacher = G_TEACHER_ID
    subject = G_SUBJECT_ID

    cur = mysql.connection.cursor()
    # if default
    # if no teacher is selected, and no subject select (no filter)
    if ((teacher == "all" or teacher == "0") and (subject == "0" or subject == "all")):
        cur.execute("SELECT AVG(neu) from evaluation WHERE score IS NOT NULL")
        neuAve = cur.fetchall()[0]
        return neuAve
    # if not default
    else:
        # if a teacher is selected, but no subject is selected (teacher + all subjects)
        if ((teacher != "0" and teacher != "all") and (subject == "0" or subject == "all")):
            sql = "SELECT AVG(neu) from evaluation where idteacher = %s and score IS NOT NULL "
            val = (teacher,)
        # if a teacher is selected, and a subject is selected (teacher + subject)
        elif ((teacher != "0" and teacher != "all") and (subject != "0" or subject != "all")):
            sql = "SELECT AVG(neu) from evaluation where idteacher = %s and edpCode = %s and score IS NOT NULL "
            val = (teacher, subject,)
        # else (if there is no teacher selected and a subject is selected)
        else:
            sql = "SELECT AVG(neu) from evaluation where edpCode = %s and score IS NOT NULL"
            val = (subject,)

        cur.execute(sql, val)
        neuAve = cur.fetchall()[0]
        return neuAve


# end for getting average for positive, negative and neutral
@app.route("/generateReport/<sec1>/<sec2>/<sec3>/<sec4>/<sec5>/<comment>/", methods=["POST", "GET"])
def generateReport(sec1, sec2, sec3, sec4, sec5, comment):
    try:
        posAve = getPositiveAverage()
        negAve = getNegativeAverage()
        neuAve = getNeutralAverage()
        resp = printReport(sec1, sec2, sec3, sec4, sec5, comment, posAve[0], negAve[0], neuAve[0])
        return resp  # anhi na part ma download ang summary report nga pdf
    except Exception as e:
        print(e)
        return "Can't print report"


# method that will send the input comment to the API and return its response
with app.app_context():
    def getsentiment(comment):
        import requests
        dictToSend = {'comment': comment}
        res = requests.post('http://127.0.0.6:8000/getSentiment', json=dictToSend)
        #res = requests.post('https://csentimentapi.herokuapp.com/getSentiment', json=dictToSend)
        print('response from server:', res.text)
        dictFromServer = res.json()
        return str(dictFromServer)

with app.app_context():
    def printReport(sec1, sec2, sec3, sec4, sec5, comment, posAve, negAve, neuAve):
        import requests
        test = {
            'Section1': sec1,
            'Section2': sec2,
            'Section3': sec3,
            'Section4': sec4,
            'Section5': sec5,
            'Comments': comment,
            'Teacher': G_TEACHER_NAME,
            'Subject': G_SUBJECT_NAME,
            'Respondents': G_NUMBER_OF_RESPONDENTS,
            'posAve': posAve,
            'negAve': negAve,
            'neuAve': neuAve
        }
        # data = [
        #     ("Section1", sec1),
        #     ("Section2", sec2),
        #     ("Section3", sec3),
        #     ("Section4", sec4),
        #     ("Section5", sec5),
        #     ("Comments", comment),
        #     ("Teacher", G_TEACHER_NAME),
        #     ("Subject", G_SUBJECT_NAME),
        #     ("Respondents", G_NUMBER_OF_RESPONDENTS),
        #     ("posAve", posAve),
        #     ("negAve", negAve),
        #     ("neuAve", neuAve),
        # ]
        data = list(test.items())
        resp = requests.post('http://127.0.0.6:8000/reportGeneration', json = test, stream=True)
        #resp = requests.post('https://csentimentapi.herokuapp.com/reportGeneration', json=data, stream=True)
        return resp.raw.read(), resp.status_code, resp.headers.items()

if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host='127.0.0.1', port=8080, debug=True)

