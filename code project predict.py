import numpy as np
import pandas as pd
df = pd.read_csv('f:/final project/colon-labled.csv')
df.head()
df.shape
#X Data
X = df.drop(['class'] , axis=1)
y = df['class']

print('X Data is \n' , X[:10])
print('X shape is ' , X.shape)
#y Data

print('y Data is \n' , y[:10])
print('y shape is ' , y.shape)
print("==========================")

print(X.describe())
# from sklearn.datasets import load_colon_cancer
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2 , f_classif 
#----------------------------------------------------
#Feature Selection by Percentile

FeatureSelection = SelectPercentile(score_func = chi2, percentile=1) # score_func can = f_classif
fit = FeatureSelection.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(20,'Score'))  #print 20 best features
X = FeatureSelection.fit_transform(X, y)

#showing X Dimension 
print('X Shape is ' , X.shape)
print('Selected Features are : ' , FeatureSelection.get_support())
print(X)
print(y)


X.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=0)
# 75% training and 25% test
print(X_train)
def models(X_train,y_train):
#using logistic regression
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(solver='liblinear', random_state=0)
    log.fit(X_train,y_train)

    #usig kneighbours classifiers
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=3,p=2)
    neigh.fit(X_train,y_train)

    #using svc linear
    from sklearn.svm import SVC
    svc_lin = SVC(kernel='linear', random_state=0)
    svc_lin.fit(X_train,y_train)


    #using dicisiontree classifier
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X_train,y_train)

    #Import Random Forest Model
    from sklearn.ensemble import RandomForestClassifier
    #import pickle
    #Create a Gaussian Classifier
    forest=RandomForestClassifier(n_estimators=12,criterion="entropy",random_state=0)

    #Train the model using the training sets y_pred=clf.predict(X_test)
    forest.fit(X_train,y_train)
# save
#print model accurancy in training data
    print("[0]logistic regression training accurancy:",log.score(X_train,y_train))
    print("[1]kneighbours classifiers training accurancy:",neigh.score(X_train,y_train))
    print("[2]svc linear training accurancy:",svc_lin.score(X_train,y_train))
    print("[4]decision tree training accurancy:",tree.score(X_train,y_train))
    print("[5]random forest training accurancy:",forest.score(X_train,y_train))
    

    return log,neigh,svc_lin,tree,tree,forest



model=models(X_train,y_train)
print(X_train[1])
print(y_train[1])

                



