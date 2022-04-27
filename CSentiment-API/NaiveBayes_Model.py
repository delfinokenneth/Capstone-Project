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

x, x_test, y, y_test = train_test_split(x,y, stratify=y, test_size=0.15, random_state=45)

# Vectorize text reviews to numbers
vec = CountVectorizer(stop_words='english')
x = vec.fit_transform(x).toarray()
x_test = vec.transform(x_test).toarray()

from sklearn.naive_bayes import MultinomialNB                                                 

model = MultinomialNB()
model.fit(x, y)

accuracy = model.score(x_test, y_test)

#print(model.predict(vec.transform(['dili kahibaw mutudlo, mura ug di kahibaw sa iyang subject'])))

#save model
pickle.dump(model, open('NB_Model.pkl', 'wb'))

#load the model
pickled_model = pickle.load(open('NB_Model.pkl', 'rb'))
comment= "yes"
print(pickled_model.predict((vec.transform([comment]))))

print(pickled_model.score(x_test,y_test))