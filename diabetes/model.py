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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
dataset = pd.read_csv('diabetes.csv')
X = dataset.iloc[:, 3:7]

y = dataset.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(X , y, test_size=0.25, random_state=0)


#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.
logisticRegr = LogisticRegression()

#Fitting model with trainig data
logisticRegr.fit(x_train, y_train)
predictions = logisticRegr.predict(x_test)
score = logisticRegr.score(x_test, y_test)
print(score)
# Saving model to disk
pickle.dump(logisticRegr, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

print(model.predict([[2, 9, 6,4]]))