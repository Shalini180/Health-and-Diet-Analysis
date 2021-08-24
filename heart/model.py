# Importing the libraries
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from pandas import Series,DataFrame
import scipy
import seaborn as sb
import sklearn
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 
dataset = pd.read_csv('heart.csv')
X = dataset.iloc[:, 0:6]

y = dataset.iloc[:, -1]
def convert_to_int(word):
    word_dict = {0:'male', 1:'female'}
    return word_dict[word]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)


#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Saving model to disk
pickle.dump(clf, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

