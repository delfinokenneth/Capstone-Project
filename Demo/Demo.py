from flask import Flask, redirect, render_template, request, url_for
from flask_mysqldb import MySQL
import platform

# create flask app
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
    app.config['MYSQL_HOST'] = 'mysql-77857-0.cloudclusters.net';
    app.config['MYSQL_USER'] = 'dbuser';
    app.config['MYSQL_PASSWORD'] = 'dbuser123';
    app.config['MYSQL_DB'] = 'isent';
    app.config['MYSQL_PORT'] = 12998;
    #old-db
    #app.config['MYSQL_HOST'] = 'mysql-76692-0.cloudclusters.net';
    #app.config['MYSQL_USER'] = 'dbuser';
    #app.config['MYSQL_PASSWORD'] = 'dbuser123';
    #app.config['MYSQL_DB'] = 'isent';
    #app.config['MYSQL_PORT'] = 14859;

# declare MySql for connection
mysql = MySQL(app)

# function to check if filters for teacher and subject is "all" meaning in default filters
def isDefaultUrlForSummary(teacher, subject):
    if teacher == "all" and subject == "all":
        return True
    else:
        return False


# get teacher name using teacher id
def getSelectedTeacherName(teacher):
    #if teacher selected is "all" return "All"
    if (teacher == "all" or teacher == "0"):
        return "All"
    # else: get the firstname and lastname of the teacher from db
    else:
        cur = mysql.connection.cursor()
        sql = "SELECT fname,lname from teachers WHERE idteacher = %s LIMIT 1"
        val = (teacher,)
        cur.execute(sql, val)
        resultTeachers = cur.fetchall()
        # resultTeachers[0][1] is the teacher's lastname and resultTeachers[0][0] for the teachers firstname
        return resultTeachers[0][1] + ", " + resultTeachers[0][0]


# get teacher name from subject id
def getSelectedSubjectTitle(teacher, subject):
    cur = mysql.connection.cursor()
    #if selected subject is all or 0, return "All"
    if (subject == "all" or subject == "0"):
        return "All"
    else:
        # if subject is 0, then get the first subject of the selected teacher
        if (subject == "0"):
            sql = "SELECT title from subjects WHERE teacherId = %s ORDER BY title asc LIMIT 1"
            val = (teacher,)
        # else get the subject
        else:
            sql = "SELECT edpcode,title from subjects WHERE edpCode = %s LIMIT 1"
            val = (subject,)
        cur.execute(sql, val)
        resultSubject = cur.fetchall()
        # return subject where resultSubject[0][0] is the edp code and resultSubject[0][1] is the subject name
        return str(resultSubject[0][0]) + "-" + resultSubject[0][1]


# get number of respondents from filtered teacher and subject
def getNumberOfRespondents(teacher, subject):
    cur = mysql.connection.cursor()
    # if default
    # if no teacher is selected, and no subject select (no filter)
    if ((teacher == "all" or teacher == "0") and (subject == "0" or subject == "all")):
        cur.execute("SELECT count(*) from evaluation INNER JOIN csentiment ON evaluation.id = csentiment.evaluationId ")
        result = cur.fetchall()
        return result[0][0]
    # if not default
    else:
        # if a teacher is selected, but no subject is selected (teacher + all subjects)
        if ((teacher != "0" or teacher != "all") and (subject == "0" or subject == "all")):
            sql = "SELECT count(*) from evaluation INNER JOIN csentiment ON evaluation.id = csentiment.evaluationId WHERE evaluation.idTeacher = %s"
            val = (teacher,)
        # else (if there is no teacher selected and a subject is selected)
        elif ((teacher == "0" or teacher == "all") and (subject != "0" or subject != "all")):
            sql = "SELECT count(*) from evaluation INNER JOIN csentiment ON evaluation.id = csentiment.evaluationId WHERE evaluation.edpCode = %s"
            val = (subject,)
        # if a teacher is selected, and a subject is selected (teacher + subject)
        elif ((teacher != "0" or teacher != "all") and (subject != "0" or subject != "all")):
            sql = "SELECT count(*) from evaluation INNER JOIN csentiment ON evaluation.id = csentiment.evaluationId WHERE evaluation.idTeacher = %s and evaluation.edpCode = %s"
            val = (teacher, subject,)


        cur.execute(sql, val)
        result = cur.fetchall()

        #return number of respondents
        return result[0][0]


# get comment,pos,neg,neu
def getSentimentValues(teacher, subject):
    cur = mysql.connection.cursor()
    # if default
    # if no teacher is selected, and no subject selected (no filter)
    # get all records where the comments is not null
    if ((teacher == "all" or teacher == "0") and (subject == "0" or subject == "all")):
        sql = "SELECT evaluation.comment,"
        sql += "csentiment.positive_value,"
        sql += "csentiment.neutral_value,"
        sql += "csentiment.negative_value,"
        sql += "csentiment.sentiment_classification,"
        sql += "csentiment.score "
        sql += "from evaluation INNER JOIN csentiment ON evaluation.id = csentiment.evaluationId "
        sql += "where evaluation.comment is not null and evaluation.comment <> ''"
        cur.execute(sql)
        return cur.fetchall()
    # if not default
    else:
        # if a teacher is selected, but no subject is selected (teacher + all subjects)
        # get all records of that teacher and its all subjects
        if ((teacher != "0" and teacher != "all") and (subject == "0" or subject == "all")):
            sql = "SELECT evaluation.comment,"
            sql += "csentiment.positive_value,"
            sql += "csentiment.neutral_value,"
            sql += "csentiment.negative_value,"
            sql += "csentiment.sentiment_classification,"
            sql += "csentiment.score "
            sql += "from evaluation INNER JOIN csentiment ON evaluation.id = csentiment.evaluationId "
            sql += "where evaluation.comment is not null and evaluation.comment <> '' and evaluation.idteacher = %s"
            val = (teacher,)
        # if a teacher is selected, and a subject is selected (teacher + subject)
        # get the records of that selected teacher and selected subject
        elif ((teacher != "0" and teacher != "all") and (subject != "0" or subject != "all")):
            sql = "SELECT evaluation.comment,"
            sql += "csentiment.positive_value,"
            sql += "csentiment.neutral_value,"
            sql += "csentiment.negative_value,"
            sql += "csentiment.sentiment_classification,"
            sql += "csentiment.score "
            sql += "from evaluation INNER JOIN csentiment ON evaluation.id = csentiment.evaluationId "
            sql += "where evaluation.comment is not null and evaluation.comment <> '' and evaluation.idteacher = %s and evaluation.edpCode = %s"
            #sql = "SELECT comment,pos,neu,neg,sentiment,score from evaluation where comment is not null and comment <> '' and idteacher = %s and edpCode = %s"
            val = (teacher, subject,)
        # else (if there is no teacher selected and a subject is selected)
        # get records of the selected subject from all teachers
        else:
            sql = "SELECT evaluation.comment,"
            sql += "csentiment.positive_value,"
            sql += "csentiment.neutral_value,"
            sql += "csentiment.negative_value,"
            sql += "csentiment.sentiment_classification,"
            sql += "csentiment.score "
            sql += "from evaluation INNER JOIN csentiment ON evaluation.id = csentiment.evaluationId "
            sql += "where evaluation.comment is not null and evaluation.comment <> '' and evaluation.edpCode = %s"
            #sql = "SELECT comment,pos,neu,neg,sentiment,score from evaluation where comment is not null and comment <> '' and edpCode = %s"
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

#endpoint for teachersevaluation (summary page)
@app.route("/teachersevaluation/<teacher>/<subject>", methods=["POST", "GET"])
def evaluate(teacher, subject):
    # create new connection
    cur = mysql.connection.cursor()
    #global declaration, meaning the value can be accessed everywhere in the class
    global G_TEACHER_NAME
    global G_SUBJECT_NAME
    global G_NUMBER_OF_RESPONDENTS
    global G_TEACHER_ID
    global G_SUBJECT_ID

    G_TEACHER_ID = teacher
    G_SUBJECT_ID = subject
    # pass to getSelectedTeacherName and get the selected teacher name
    # TO FURTHER UNDERSTAND THE NEXT 3 LINES, CHECK ITS FUNCTIONS
    G_TEACHER_NAME = getSelectedTeacherName(teacher)
    # pass to getSelectedSubjectTitle and get the selected subject title
    G_SUBJECT_NAME = getSelectedSubjectTitle(teacher, subject)
    # pass to getNumberOfRespondents and get the number of respondents based on the filters
    G_NUMBER_OF_RESPONDENTS = getNumberOfRespondents(teacher, subject)

    # common queries
    #get all questionaire of section 1
    cur.execute("SELECT * FROM questionaire where section = 1")
    section1 = cur.fetchall()
    #get all questionaire of section 2
    cur.execute("SELECT * FROM questionaire where section = 2")
    section2 = cur.fetchall()
    #get all questionaire of section 3
    cur.execute("SELECT * FROM questionaire where section = 3")
    section3 = cur.fetchall()
    #get all questionaire of section 4
    cur.execute("SELECT * FROM questionaire where section = 4")
    section4 = cur.fetchall()
    #get all questionaire of section 5
    cur.execute("SELECT * FROM questionaire where section = 5")
    section5 = cur.fetchall()

    # ask kenneth in this part
    # <!-- DB guide-> https://imgur.com/YMKA4ib -->
    cur.execute("""SELECT DISTINCT section.id, section.section, section.name, section.description, section.percentage, 
				(select count(question) from questionaire  where section = '1') as total1, 
				(select count(question) from questionaire  where section = '2') as total2, 
				(select count(question) from questionaire  where section = '3') as total3, 
				(select count(question) from questionaire  where section = '4') as total4,
				(select count(question) from questionaire  where section = '5') as total5 
				from section 
				right join questionaire on section.section = questionaire.section """)

    #fetch result/data from the database of the previous query
    sectionsleft = cur.fetchall()

    # get the questionaire section and question from the database
    cur.execute(""" SELECT questionaire.section, questionaire.question from questionaire
					right join section
					ON questionaire.section = section.section """)
    sectionsright = cur.fetchall()
    # End for common queries

    # FOR DEFAULT QUERIES
    # queries for the teachers and subject filter
    # get teachers from query, selected nga teacher should be on top
    sql = "SELECT * FROM teachers order by (case idteacher when %s then 0 else 1 end), lname asc"
    val = (teacher,)
    cur.execute(sql, val)
    teachers = cur.fetchall()

    #if a teacher is select but subject is in default = 0, 1st subject of the selected teacher should be on top
    if (subject != "all" and (teacher != "0")):
        sql = "SELECT * FROM subjects WHERE teacherId = %s or edpCode = 0 order by (case edpCode when %s then 0 else 1 end), title asc"
        val = (teachers[0][0], subject,)
        cur.execute(sql, val)
        subjects = cur.fetchall()
    # else if a subject is selected, selected subject should be on top
    else:
        sql = "SELECT * FROM subjects order by (case edpCode when %s then 0 else 1 end), title asc"
        val = (subject,)
        cur.execute(sql, val)
        subjects = cur.fetchall()
    # end for default queries

    # get sentiment values
    # to understand this line, visit getSentimentValues() Function
    comments = getSentimentValues(teacher, subject)

    # get total number of respondents
    # to understand this line, visit getNumberOfRespondents() Function
    numofrespondents = getNumberOfRespondents(teacher, subject)

    # get rating records from all sections
    # to understand this line, visit getRatingValues() Function
    evalsecans = getRatingValues(teacher, subject)

    # END FOR FILTER RECORDS
    cur.close()

    # then mugawas na dayon tong summary nga page, and gipasa ang mga values (refer below)
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

# this the for the evaluation nga page (katong pag evaluate)
@app.route("/evaluation/<teacher>/<subject>", methods=["POST", "GET"])
def evaluation(teacher, subject):
    #fetch all questions from nga section 1
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM questionaire where section = 1")
    section1 = cur.fetchall()

    #fetch all questions from nga section 1
    cur.execute("SELECT * FROM questionaire where section = 2")
    section2 = cur.fetchall()

    #fetch all questions from nga section 1
    cur.execute("SELECT * FROM questionaire where section = 3")
    section3 = cur.fetchall()

    #fetch all questions from nga section 1
    cur.execute("SELECT * FROM questionaire where section = 4")
    section4 = cur.fetchall()

    #fetch all questions from nga section 1
    cur.execute("SELECT * FROM questionaire where section = 5")
    section5 = cur.fetchall()

    # FOR DEFAULT QUERIES
    # queries for the teachers and subject filter
    # para ni atong dropdown button
    # fetch all teacher, and selected teacher should be on top sa dropdown
    sql = "SELECT * FROM teachers order by (case idteacher when %s then 0 else 1 end), lname asc"
    val = (teacher,)
    cur.execute(sql, val)
    teachers = cur.fetchall()

     # fetch all teacher, and selected teacher should be on top sa dropdown
     # if there is selected subject, selected subject should be on top
    sql = "SELECT * FROM subjects WHERE teacherId = %s or edpCode = 0 order by (case edpCode when %s then 0 else 1 end), title asc"
    val = (teachers[0][0], subject,)
    cur.execute(sql, val)
    subjects = cur.fetchall()
    # end for default queries

    cur.close()

    #if musubmit ang user sa evaluation, ma trigger ni nga part
    if request.method == 'POST':
        # Declaring variables for list to store rating in each section
        sec1_rating = []
        sec2_rating = []
        sec3_rating = []
        sec4_rating = []
        sec5_rating = []

        #store all the ratings from section 1 questions into the sec1_rating list variable
        for i in range(len(section1)):
            sec1_rating.append(request.form[f'rating[{i}]'])

        #store all the ratings from section 2 questions into the sec1_rating list variable
        for i in range(len(section2)):
            sec2_rating.append(request.form[f'rating2[{i}]'])

        #store all the ratings from section 3 questions into the sec1_rating list variable
        for i in range(len(section3)):
            sec3_rating.append(request.form[f'rating3[{i}]'])

        #store all the ratings from section 4 questions into the sec1_rating list variable
        for i in range(len(section4)):
            sec4_rating.append(request.form[f'rating4[{i}]'])

        #store all the ratings from section 5 questions into the sec1_rating list variable
        for i in range(len(section5)):
            sec5_rating.append(request.form[f'rating5[{i}]'])

        # code for the translation and getting sentiment analysis
        comment = request.form["txtcomment"]
        comment = comment.replace("miss", "")

        # getting the sentiment and details from API
        # getsentiment() function is mao ni gigamit pagkuhas sentiment
        # PLEASE VISIT THIS FUNCTION TO UNDERSTAND 

        # index[1] = get the positive value from the response from API
        pos_val = getsentiment(comment).split(" ")[1]
        # index[2] = get the neutral value from the response from API
        neu_val = getsentiment(comment).split(" ")[2]
        # index[3] = get the negative value from the response from API
        neg_val = getsentiment(comment).split(" ")[3]
        # index[4] = get the score value from the response from API
        score_val = getsentiment(comment).split(" ")[4]
        # index[0] = get the classification value from the response from API
        sen_val = getsentiment(comment).split(" ")[0]
        # if sentiment is neutral then score = null

        # if the classification is neutral, Null is save to database
        if sen_val == 'neutral':
            score_val = None

        try:
            #open the database
            cur = mysql.connection.cursor()
            # converting list into string para maoy isave sa database
            sec1_string = ','.join(sec1_rating)
            sec2_string = ','.join(sec2_rating)
            sec3_string = ','.join(sec3_rating)
            sec4_string = ','.join(sec4_rating)
            sec5_string = ','.join(sec5_rating)

            # if input comment is not empty
            if comment is not "":
                # if subject is 0 or default, get the value of first subject
                if (subject == "0" and subject != "all"):
                    #set subject to first subject
                    subject = subjects[0][0]
                # insert data into database
                sql = "INSERT INTO evaluation (idteacher,idstudent,edpCode,section1,section2,section3,section4,section5,pos,neu,neg,comment,sentiment, score)\
					 VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);"

                #values to save
                val = (
                teacher, "18013672", subject, sec1_string, sec2_string, sec3_string, sec4_string, sec5_string, pos_val,
                neu_val, neg_val, comment, sen_val, score_val)

                # getting the last row id inserted in evaluation table
                cur.execute(sql, val)
                # commit changes
                mysql.connection.commit()
                
                # get the last id of the evaluation table
                id = cur.lastrowid
                # inserting sentiment values to table csentiment
                sql = "INSERT INTO csentiment (evaluationId,comments,positive_value,neutral_value,negative_value,sentiment_classification,score)\
                                VALUES (%s,%s,%s,%s,%s,%s,%s);"
                val = (id,comment, pos_val,neu_val,neg_val,sen_val, score_val)
                
                
            # else input comment is empty
            else:
                sql = "INSERT INTO evaluation (idteacher,idstudent,edpCode,section1,section2,section3,section4,section5)\
							 VALUES (%s,%s,%s,%s,%s,%s,%s,%s);"
                val = (teacher, "18013672", subject, sec1_string, sec2_string, sec3_string, sec4_string, sec5_string)
                # getting the last row id inserted in evaluation table
                cur.execute(sql, val)
                mysql.connection.commit()
                # get the last id of the evaluation table
                id = cur.lastrowid
                # inserting sentiment values to table csentiment
                sql = "INSERT INTO csentiment (evaluationId,comments,positive_value,neutral_value,negative_value,sentiment_classification,score)\
                                VALUES (%s,%s,%s,%s,%s,%s,%s);"
                val = (id, comment, pos_val, neu_val, neg_val, sen_val, score_val)

            cur.execute(sql, val)
            mysql.connection.commit()
            cur.close()

            #after masave ang gi evaluation, mu redirect adtos summary nga screen
            return redirect("/teachersevaluation/all/all")

        except Exception as exp:
            return f'<h1>{exp}</h1>'

    # wala nagsubmit ang user, muredirect ras evaluation page
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
        cur.execute("SELECT AVG(positive_value) from csentiment WHERE score IS NOT NULL ")
        posAve = cur.fetchall()[0]
        return posAve
    # if not default
    else:
        # if a teacher is selected, but no subject is selected (teacher + all subjects)
        if ((teacher != "0" and teacher != "all") and (subject == "0" or subject == "all")):
            sql = "SELECT AVG(csentiment.positive_value) "
            sql += "from csentiment INNER JOIN evaluation ON "
            sql += "csentiment.evaluationId = evaluation.id "
            sql += "where evaluation.idteacher = %s and csentiment.score IS NOT NULL "
            val = (teacher,)
        # if a teacher is selected, and a subject is selected (teacher + subject)
        elif ((teacher != "0" and teacher != "all") and (subject != "0" or subject != "all")):
            sql = "SELECT AVG(csentiment.positive_value) "
            sql += "from csentiment INNER JOIN evaluation ON "
            sql += "csentiment.evaluationId = evaluation.id "
            sql += "where evaluation.idteacher = %s and evaluation.edpCode = %s and csentiment.score IS NOT NULL "
            #sql = "SELECT AVG(pos) from evaluation where idteacher = %s and edpCode = %s and score IS NOT NULL "
            val = (teacher, subject,)
        # else (if there is no teacher selected and a subject is selected)
        else:
            sql = "SELECT AVG(csentiment.positive_value) "
            sql += "from csentiment INNER JOIN evaluation ON "
            sql += "csentiment.evaluationId = evaluation.id "
            sql += "where evaluation.edpCode = %s and csentiment.score IS NOT NULL "
            #sql = "SELECT AVG(pos) from evaluation where edpCode = %s and score IS NOT NULL "
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
        cur.execute("SELECT AVG(csentiment.negative_value) from csentiment WHERE score IS NOT NULL ")
        posAve = cur.fetchall()[0]
        return posAve
    # if not default
    else:
        # if a teacher is selected, but no subject is selected (teacher + all subjects)
        if ((teacher != "0" and teacher != "all") and (subject == "0" or subject == "all")):
            sql = "SELECT AVG(csentiment.negative_value) "
            sql += "from csentiment INNER JOIN evaluation ON "
            sql += "csentiment.evaluationId = evaluation.id "
            sql += "where evaluation.idteacher = %s and csentiment.score IS NOT NULL "
            val = (teacher,)
        # if a teacher is selected, and a subject is selected (teacher + subject)
        elif ((teacher != "0" and teacher != "all") and (subject != "0" or subject != "all")):
            sql = "SELECT AVG(csentiment.negative_value) "
            sql += "from csentiment INNER JOIN evaluation ON "
            sql += "csentiment.evaluationId = evaluation.id "
            sql += "where evaluation.idteacher = %s and evaluation.edpCode = %s and csentiment.score IS NOT NULL "
            #sql = "SELECT AVG(pos) from evaluation where idteacher = %s and edpCode = %s and score IS NOT NULL "
            val = (teacher, subject,)
        # else (if there is no teacher selected and a subject is selected)
        else:
            sql = "SELECT AVG(csentiment.negative_value) "
            sql += "from csentiment INNER JOIN evaluation ON "
            sql += "csentiment.evaluationId = evaluation.id "
            sql += "where evaluation.edpCode = %s and csentiment.score IS NOT NULL "
            #sql = "SELECT AVG(pos) from evaluation where edpCode = %s and score IS NOT NULL "
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
        cur.execute("SELECT AVG(csentiment.neutral_value) from csentiment WHERE score IS NOT NULL ")
        posAve = cur.fetchall()[0]
        return posAve
    # if not default
    else:
        # if a teacher is selected, but no subject is selected (teacher + all subjects)
        if ((teacher != "0" and teacher != "all") and (subject == "0" or subject == "all")):
            sql = "SELECT AVG(csentiment.neutral_value) "
            sql += "from csentiment INNER JOIN evaluation ON "
            sql += "csentiment.evaluationId = evaluation.id "
            sql += "where evaluation.idteacher = %s and csentiment.score IS NOT NULL "
            val = (teacher,)
        # if a teacher is selected, and a subject is selected (teacher + subject)
        elif ((teacher != "0" and teacher != "all") and (subject != "0" or subject != "all")):
            sql = "SELECT AVG(csentiment.neutral_value) "
            sql += "from csentiment INNER JOIN evaluation ON "
            sql += "csentiment.evaluationId = evaluation.id "
            sql += "where evaluation.idteacher = %s and evaluation.edpCode = %s and csentiment.score IS NOT NULL "
            #sql = "SELECT AVG(pos) from evaluation where idteacher = %s and edpCode = %s and score IS NOT NULL "
            val = (teacher, subject,)
        # else (if there is no teacher selected and a subject is selected)
        else:
            sql = "SELECT AVG(csentiment.neutral_value) "
            sql += "from csentiment INNER JOIN evaluation ON "
            sql += "csentiment.evaluationId = evaluation.id "
            sql += "where evaluation.edpCode = %s and csentiment.score IS NOT NULL "
            #sql = "SELECT AVG(pos) from evaluation where edpCode = %s and score IS NOT NULL "
            val = (subject,)

    cur.execute(sql, val)
    neuAve = cur.fetchall()[0]
    return neuAve


# end for getting average for positive, negative and neutral
@app.route("/generateReport/<sec1>/<sec2>/<sec3>/<sec4>/<sec5>/<comment>/<ratingPerc>/<commentPerc>/", methods=["POST", "GET"])
def generateReport(sec1, sec2, sec3, sec4, sec5, comment, ratingPerc, commentPerc):
    try:
        #get the average of positive scores
        #to further understang this functions visit getPositiveAverage() function
        posAve = getPositiveAverage()
        #get the average of negative scores
        #to further understang this functions visit getNegativeAverage() function
        negAve = getNegativeAverage()
        #get the average of neutral scores
        #to further understang this functions visit getNeutralAverage() function
        neuAve = getNeutralAverage()
        # get the response from API to download the PDF file
        # visit printReport() function to para makita giunsa pag connect sa API
        resp = printReport(sec1, sec2, sec3, sec4, sec5, comment, posAve[0], negAve[0], neuAve[0], ratingPerc, commentPerc)
        return resp  # anhi na part ma download ang summary report nga pdf
    except Exception as e:
        print(e)
        return "Can't print report"


# method that will send the input comment to the API and return its response
# This are the part ni connect and Demo.py sa API
with app.app_context():
    #this is for the sentimentAnalyzer
    def getsentiment(comment):
        import requests
        dictToSend = {'comment': comment}
        #mao ni pag call sa API and the response from APi is stored in res nga variable
        res = requests.post('http://127.0.0.6:8000/getSentiment', json=dictToSend)
        #res = requests.post('https://csentiment-api.herokuapp.com/getSentiment', json=dictToSend)
        print('response from server:', res.text)
        #convert the response into json
        dictFromServer = res.json()
        return str(dictFromServer)

with app.app_context():
    #this is for the report Generation
    def printReport(sec1, sec2, sec3, sec4, sec5, comment, posAve, negAve, neuAve, ratingPerc, commentPerc):
        import requests
        # this data should be pass to API for creating the PDF file 
        # if kuwang/sobra nag data, it will error and can't generate file
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
            'neuAve': neuAve,
            'ratingPercentage': ratingPerc,
            'commentPercentage': commentPerc,
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
        #mao ni ang pagcall sa API for report generation and response is store in resp nga variable
        resp = requests.post('http://127.0.0.6:8000/reportGeneration', json = test, stream=True)
        #resp = requests.post('https://csentimentapi.herokuapp.com/reportGeneration', json=data, stream=True)
        return resp.raw.read(), resp.status_code, resp.headers.items()

if __name__ == "__main__":
    app.run(debug=True)
    #app.run(host='127.0.0.1', port=8080, debug=True)

