# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 06:51:19 2019

@author: sai
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import the excel file
dataset = pd.read_excel('credit_card.xls')
x = dataset.iloc[1: , 0:-1].values
y = dataset.iloc[1: , -1].values
np.set_printoptions(threshold = np.nan)
from sklearn.impute import SimpleImputer
#replace missing values with mean
simpleimputer = SimpleImputer(np.nan,"mean")
simpleimputer = simpleimputer.fit(x[:,:])

x[:,:] = simpleimputer.transform(x[:,:])

"""from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(categories = "auto"), # The transformer class
         [0]           # The column(s) to be applied on.
         )
    ]
    ,remainder='passthrough' # not mentioned columns are allowed to passthrough if 'drop' is mentioned instead of
    #of 'passthrough' then not mentioned coloumns will be droped
)
x = transformer.fit_transform(x)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)"""

from sklearn.model_selection import train_test_split 
#split the data into test and train where test size = 25% and train size = 80%
x_train , x_test , y_train ,y_test = train_test_split(x,y,test_size = 0.25,random_state = 0)
y_train=y_train.astype('int')  
y_test=y_test.astype('int')  

from sklearn.preprocessing import StandardScaler
#perform standardadization on the test and train sample (x - mean) / sd
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier  #import knn classifier

classifier = KNeighborsClassifier(n_neighbors = 5,metric = 'minkowski', p = 2)
#knn classifier
classifier.fit(x_train,y_train)
#classifier is fit onto our training set and the classifier will be able to learn the co-relation


y_pred = classifier.predict(x_test) #predict the test set results


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred) # making confusion matrix

#accuracy calculations for id3 algorithm
knnaccuracy = (cm[0][0] + cm[1][1]) * 100/(cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])

print('\ntotal accuracy of knn algorithm : ')
print(knnaccuracy)

knnfraudaccuracy = (cm[1][1])*100/(cm[1][0] + cm[1][1])
print('\naccuracy of predicting default :')
print(knnfraudaccuracy)

knn_notfraudaccuracy = (cm[0][0])*100/(cm[0][0] + cm[0][1])
print('\naccuracy of predicting not a default :')
print(knn_notfraudaccuracy)

print('\n\n\n')

from sklearn import tree

classifier1 = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, 
                            min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                            max_features=None, random_state=0, max_leaf_nodes=None,
                            min_impurity_decrease=0.0, min_impurity_split=None,
                            class_weight=None, presort=False)  # cart classifier

classifier1.fit(x_train,y_train)
#classifier is fit onto our training set and the classifier will be able to learn the co-relation


y_pred1 = classifier1.predict(x_test)  #predict the test set results

#making the confusion matrix
from sklearn.metrics import confusion_matrix

cm1 = confusion_matrix(y_test,y_pred1) #making confusion matrix

#accuracy calculations for id3 algorithm
cartaccuracy = (cm1[0][0] + cm1[1][1]) * 100/(cm1[0][0] + cm1[0][1] + cm1[1][0] + cm1[1][1])

print('\ntotal accuracy of cart algorithm : ')
print(cartaccuracy)

cartfraudaccuracy = (cm1[1][1])*100/(cm1[1][0] + cm1[1][1])
print('\naccuracy of predicting default :')
print(cartfraudaccuracy)

cartnotfraudaccuracy = (cm1[0][0])*100/(cm1[0][0] + cm1[0][1])
print('\naccuracy of predicting not a default :')
print(cartnotfraudaccuracy)+

print('\n\n\n')



#logistic regression

from sklearn.linear_model import LogisticRegression   #import logistic regression classifier

classifier2 = LogisticRegression(random_state = 0)
# logistic regression classifier
classifier2.fit(x_train,y_train)
#classifier is fit onto our training set and the classifier will be able to learn the co-relation

y_pred2 = classifier2.predict(x_test) #predict the test set results

from sklearn.metrics import confusion_matrix

cm2 = confusion_matrix(y_test,y_pred2) #making confusion matrix

#accuracy calculations for logistic regression algorithm
lraccuracy = (cm2[0][0] + cm2[1][1]) * 100/(cm2[0][0] + cm2[0][1] + cm2[1][0] + cm2[1][1])

print('\ntotal accuracy of logistic regression algorithm : ')
print(lraccuracy)

lrfraudaccuracy = (cm2[1][1])*100/(cm2[1][0] + cm2[1][1])
print('\naccuracy of predicting default :')
print(lrfraudaccuracy)

lrnotfraudaccuracy = (cm2[0][0])*100/(cm2[0][0] + cm2[0][1])
print('\naccuracy of predicting not a default :')
print(lrnotfraudaccuracy)

print('\n\n\n')

#Decision tree id3

from sklearn.tree import DecisionTreeClassifier

classifier4 = DecisionTreeClassifier(criterion = 'entropy')  #id3 classifier

classifier4.fit(x_train,y_train)
#classifier is fit onto our training set and the classifier will be able to learn the co-relation


y_pred4 = classifier4.predict(x_test)  #predict the test set results

from sklearn.metrics import confusion_matrix

cm3 = confusion_matrix(y_test,y_pred4) #making confusion matrix

#accuracy calculations for id3 algorithm
id3accuracy = (cm3[0][0] + cm3[1][1]) * 100/(cm3[0][0] + cm3[0][1] + cm3[1][0] + cm3[1][1])

print('\ntotal accuracy of id3 algorithm : ')
print(id3accuracy)

id3fraudaccuracy = (cm3[1][1])*100/(cm3[1][0] + cm3[1][1])
print('\naccuracy of predicting default :')
print(id3fraudaccuracy)

id3notfraudaccuracy = (cm3[0][0])*100/(cm3[0][0] + cm3[0][1])
print('\naccuracy of predicting not a default :')
print(id3notfraudaccuracy)

print('\n\n\n')