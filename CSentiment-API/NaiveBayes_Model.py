import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import pickle

data = pd.read_csv('Comments.csv')

data.head()

def preprocess_data(data):
    # Remove package name as it's not relevant
    data = data.drop('language', axis=1)
    
    # Convert text to lowercase
    data['comment'] = data['comment'].str.strip().str.lower()
    return data

data = preprocess_data(data)

# Split into training and testing data
x = data['comment']
y = data['label']

x, x_test, y, y_test = train_test_split(x,y, stratify=y, test_size=0.30, random_state=50)

# Vectorize text reviews to numbers
vec = CountVectorizer(stop_words='english')
x = vec.fit_transform(x).toarray()
#test_vectors = vec.transform(x_test)
#print(test_vectors)
x_test = vec.transform(x_test).toarray()
print(x_test)

from sklearn.naive_bayes import MultinomialNB                                                 

model = MultinomialNB()
model.fit(x, y)

from sklearn import metrics
from sklearn.metrics import accuracy_score

predicted_train = model.predict(x)
print("Training Accuracy ", accuracy_score(y,predicted_train))
print(metrics.classification_report(y,predicted_train))
metrics.confusion_matrix(y,predicted_train)

predicted_test = model.predict(x_test)
print("Testing Accuracy ", accuracy_score(y_test,predicted_test))
print(metrics.classification_report(y_test,predicted_test))
metrics.confusion_matrix(y_test,predicted_test)

#save model
pickle.dump(model, open('NB_Model.pkl', 'wb'))

#load the model
pickled_model = pickle.load(open('NB_Model.pkl', 'rb'))
comment= "yes"
print(pickled_model.predict((vec.transform([comment]))))

print(pickled_model.score(x_test,y_test))