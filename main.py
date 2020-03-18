import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


#to see all of the columns instead of .... in the middle
pd.set_option('display.expand_frame_repr', False)

#Step 1: Collecting Data
titanic_data = pd.read_csv('titanic.csv')
#print('# of passengers in original data: ' + str(len(titanic_data.index)))

#Step 2: Analyzing Data
#plots how many people survived and didn't survive
#sns.countplot(x='survived', data=titanic_data)
#plt.show()

#plots how many men and women survived and how many men and women didn't survive
#sns.countplot(x='survived', hue='sex',data=titanic_data)
#plt.show()

#plots how people survived based on class (first, second, third)
#sns.countplot(x='survived', hue='pclass',data=titanic_data)
#plt.show()

#plot to show the age distribution of the passengers
#titanic_data['age'].plot.hist()

#plot to show the fare price distribution
#titanic_data['fare'].plot.hist()
#titanic_data.info()

#plot number of siblings for each passenger
#sns.countplot(x="sibsp", data=titanic_data)
#plt.show()

#Step 3: Data Wrangling (cleaning the data)
#print(titanic_data.isnull()) #this tells what values are null in dataset
#print(titanic_data.isnull().sum()) #this tells what values are null in dataset based on the columns

#plots to show the age of the passengers in the different classes
#sns.boxplot(x='pclass', y='age', data=titanic_data)
#plt.show()

#in order to remove a column. In this case the cabin
#titanic_data.drop('cabin', axis=1, inplace=True)

#to drop last row in dataframe
titanic_data = titanic_data[:-1]

#in order to replace all null values with 0
titanic_data = titanic_data.replace(np.nan, 0)

#in order to convert sex string values to 0 and 1
sex = pd.get_dummies(titanic_data['sex'], drop_first=True)

embarked = pd.get_dummies(titanic_data['embarked'], drop_first=True)

pcl = pd.get_dummies(titanic_data['pclass'], drop_first=True)

#to concatenate all these new rows to the dataset
titanic_data=pd.concat([titanic_data, sex, embarked, pcl], axis=1)

#now drop the old columns which you transformed + unessessary columns
titanic_data.drop(['sex', 'pclass', 'embarked', 'name', 'ticket', 'cabin', 'boat', 'body', 'home.dest'], axis=1, inplace=True)

#Step 4: Train & Test dataset
#x will be all of the columns except for the column we are trying to predict
X= titanic_data.drop('survived', axis=1)
#y is the target column which we are trying to predict
y= titanic_data['survived']

#split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

#using Logistic Regression instance for the predictions
logmodel = LogisticRegression(max_iter=2000)
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)

#Step 5: Accuracy Check
#to evaluate how the model has been performing
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print('Accuracy score is: ' + str(accuracy_score(y_test, predictions)*100) + '%')