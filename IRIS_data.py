# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:01:23 2020

@author: Mayank Joshi
"""

##loading important packages
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_palette('husl')
import warnings
warnings.filterwarnings("ignore")

##specifying path dir
path = r'C:\Users\mayank joshi\Desktop\Padhai Likhai\ML Projects\Iris Data'
os.chdir(path)

##loading data
iris = pd.read_csv('IRIS.csv')

##taking a look at the data
iris.head(10)
iris.info()
data_description = iris.describe()
iris['species'].value_counts()

#iris.groupby('species').describe().unstack()

klmns = iris.columns

for i in klmns[0:4]:
    print(i)
    print(iris.groupby('species')[i].describe())
    print("________________________")
    print("               ")
    
##---------------------------EDA-------------------------------------
##correlation table
corr = iris.corr()
cov = iris.cov()
##correlation matrix
sns.heatmap(iris.corr(), annot=True, cmap = 'viridis')
#pairplot with hue= species
plot = sns.pairplot(iris, hue = 'species')

##histogram
iris.plot.hist(subplots=True, layout=(2,2), figsize=(10, 10), bins=20)
plt.show()
##scatter plot with hue
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=iris)
plt.show()

sns.scatterplot(x='petal_length', y='petal_width', hue='species', data=iris)
plt.show()


##boxplot
# Grouped boxplot
##Grouped boxplot are used when you have a numerical variable, several groups and subgroups.
#It is easy to realize one using seaborn.Y is your numerical variable, x is the group column,
# and hue is the subgroup column.

##grid of boxplots

plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.boxplot(x='species',y='petal_length',data=iris)
plt.subplot(2,2,2)
sns.boxplot(x='species',y='petal_width',data=iris)
plt.subplot(2,2,3)
sns.boxplot(x='species',y='sepal_length',data=iris)
plt.subplot(2,2,4)
sns.boxplot(x='species',y='sepal_width',data=iris)
plt.show()

##-----------------------------------------------------------------
##grid of violin plot

plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='species',y='petal_length',data=iris)
plt.subplot(2,2,2)
sns.violinplot(x='species',y='petal_width',data=iris)
plt.subplot(2,2,3)
sns.violinplot(x='species',y='sepal_length',data=iris)
plt.subplot(2,2,4)
sns.violinplot(x='species',y='sepal_width',data=iris)
plt.show()

## modelling 

##Split the training and test dataset

X = iris.drop(['species'], axis=1)
y = iris['species']
# print(X.head())
print(X.shape)
# print(y.head())
print(y.shape)

#Advantages
#By splitting the dataset pseudo-randomly into a two separate sets, 
#we can train using one set and test using another.
#This ensures that we won't use the same observations in both sets. 
#More flexible and faster than creating a model using all of the dataset for training.
#Disadvantages
#The accuracy scores for the testing set can vary depending on what observations are in the set.
#This disadvantage can be countered using k-fold cross-validation.
#Notes
#The accuracy score of the models depends on the observations in the testing set,
# which is determined by the seed of the pseudo-random number generator (random_state parameter).
#As a model's complexity increases, the training accuracy (accuracy you get when you train and test 
#the model on the same data) increases.
#If a model is too complex or not complex enough, the testing accuracy is lower.
#For KNN models, the value of k determines the level of complexity. A lower value of k means that 
#the model is more complex.

from sklearn.model_selection import train_test_split  #to split the dataset for training and testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , random_state=6)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#Classification Algorithms: which we shall use with this IRIS (structured) dataset
#Logistic Regression
#Decision Tree
#Support Vector Machine (SVM)
#K-Nearest Neighbours

##-----------------------------------------------------------------
# Importing alll the necessary packages to use the various classification algorithms

from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import metrics #for checking the model accuracy
from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn.metrics import classification_report ##to get classification report

##-----------------------------------------------------------------

##function to get the accurasy measures and classification report

def model_performance(actual,predicted):
    count_misclassified = (actual != predicted).sum()
    print('Misclassified samples: {}'.format(count_misclassified))
    
    acc_log = metrics.accuracy_score(predicted , actual)
    print('Accuracy: {:.2f}'.format(acc_log))
    
    results = confusion_matrix(actual, predicted)
    print ('Confusion Matrix :')
    print(results) 
    
    print ('Report : ')
    print (classification_report(actual, predicted) )

##Logistic Regression
#
#logr = LogisticRegression()
#logr.fit(X_train,y_train)
#y_pred = logr.predict(X_test)
#model_performance(y_test,y_pred)
#print('___________________________________________')
#print('                                           ')
###-----------------------------------------------------------------
###Decision Tree
#dt = DecisionTreeClassifier()
#dt.fit(X_train,y_train)
#y_pred = dt.predict(X_test)
## how did our model perform?
#model_performance(y_test,y_pred)
#print('___________________________________________')
#print('                                           ')
#
#
###------------------------------------------------------------------
###Support Vector Machine
#sv = svm.SVC() #select the algorithm
#sv.fit(X_train,y_train) # we train the algorithm with the training data and the training output
#y_pred = sv.predict(X_test) #now we pass the testing data to the trained algorithm
#
## how did our model perform?
#model_performance(y_test,y_pred)
#print('___________________________________________')
#print('                                           ')
###-----------------------------------------------------------------
###KNN
#knc = KNeighborsClassifier(n_neighbors= 3) 
##this examines 3 neighbours for putting the new data into a class
#knc.fit(X_train,y_train)
#y_pred = knc.predict(X_test)
#
## how did our model perform?
#model_performance(y_test,y_pred)
#print('___________________________________________')
#print('                                           ')

#Let's check the accuracy for various values of n for K-Nearest nerighbours
a_index = list(range(1,11))
a = pd.Series()
x = [1,2,3,4,5,6,7,8,9,10]
for i in list(range(1,11)):
    kcs = KNeighborsClassifier(n_neighbors=i) 
    kcs.fit(X_train,y_train)
    y_pred = kcs.predict(X_test)
    print(i)
    print(metrics.accuracy_score(y_pred,y_test))
    a=a.append(pd.Series(metrics.accuracy_score(y_pred,y_test)))
    
plt.plot(a_index, a)
plt.xticks(x)
plt.show()

## consolidating accuracy scores
#models = pd.DataFrame({
#    'Model': ['Logistic Regression', 'Decision Tree', 'Support Vector Machines',
#              'K-Nearest Neighbours'],
#    'Score': [acc_log, acc_dt, acc_svm, acc_knn]})
#models.sort_values(by='Score', ascending=False)


##-----------------
models = pd.DataFrame(columns=['Model_name','Model', 'Accuracy_Score'])
logr = LogisticRegression()
dt = DecisionTreeClassifier()
sv = svm.SVC() #select the algorithm
knc = KNeighborsClassifier(n_neighbors= 5) 

models = models.append({'Model_name' : 'Logistic Regression','Model': logr, 'Accuracy_Score' : ""},ignore_index=True)
models = models.append({'Model_name' : 'Decision Tree','Model': dt, 'Accuracy_Score' : ""},ignore_index=True)
models = models.append({'Model_name' : 'SVM','Model': sv, 'Accuracy_Score' : ""},ignore_index=True)
models = models.append({'Model_name' : 'KNN','Model': knc, 'Accuracy_Score' : ""},ignore_index=True)

for model in range(0,models.shape[0]):
    print('-----------------------  '+models['Model_name'][model]+'  -----------------------------')
    print('                                                                  ')
    models['Model'][model].fit(X_train,y_train)
    y_pred = models['Model'][model].predict(X_test)
    # how did our model perform?
    acc_log = metrics.accuracy_score(y_test , y_pred)
    models['Accuracy_Score'][model] = acc_log
    model_performance(y_test,y_pred)
    print('___________________________________________')
    print('                                           ')

print(models)
