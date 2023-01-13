# import personal modules
import prepare as prep
#import acquire as ac
#import datascience libraries
import pandas as pd
import numpy as np

# Sklearn modules including classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier  # Gradient Boosting Classifier
from sklearn.ensemble import HistGradientBoostingClassifier # Sklearn version of LGBM Classifier
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes Classifier


# Sklearn testing, evaluating, and managing model
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, f_regression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score


# additional, advanced classifiers
from xgboost import XGBClassifier as xgb  # XG Boost Classifier
from lightgbm import LGBMClassifier # Light Gradient Boost Classifier
from catboost import CatBoostClassifier # Cat boost classifier


# import modules from standard library
from time import time
from pprint import pprint # pretty print
from importlib import reload
import os


# NLP related modules / libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk #Natural Language Tool Kit
import re   #Regular Expressions

np.random.seed(7)




def get_model_tests():
    NB_training()
    gb_training()
    xgb_training()


def get_df():
    
    if os.path.isfile('prepared_data.csv'):
        return pd.read_csv('prepared_data.csv', index_col=[0])
    else:
        df = pd.read_csv('clean_scraped_data.csv', index_col=[0])
        df = prep.map_other_languages(df)
        
        df.to_csv('prepared_data.csv', index=False)
        
        return df


def get_xy():
    df = get_df()
        
    x = df['lemmatized']
    y = df['language']

    cv = CountVectorizer()
    x_vectorized = cv.fit_transform(x)

    
    
    return x_vectorized, y


def get_split_data():
    
    df = get_df()
    
    x = df['lemmatized']
    y = df['language']

    cv = CountVectorizer()
    #x_vectorized = cv.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 7)
    
    x_train = cv.fit_transform(x_train)
    x_test = cv.transform(x_test)
    
    
    return x_train, y_train, x_test, y_test



#########################################################
############         Model Collection      ##############
  ######  Models used with hyperparameter tuning  ######
########################################################

#########################################################################
           ############       Random Forest       ##############     
  ######  Creates N number of trees using random starting values  ######
########################################################################

def random_forest_model(x, y):
    
    rf_classifier = RandomForestClassifier(
        min_samples_leaf=10,
        n_estimators=200,
        max_depth=5, 
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        max_features='auto'
    )

    rf_classifier.fit(x, y)

    y_preds = rf_classifier.predict(x)
    
    return y_preds


#############################################################################
    ############       Gradient Boosting Classifier       ##############     
######  Creates a random forest where each tree learns from the last  ######
############################################################################

def gradient_booster_model(x_train, y_train, x_test = 0, y_test = 0, test = False):

    gradient_booster = GradientBoostingClassifier(
                            learning_rate=0.1,
                            max_depth = 5,
                            n_estimators=200)
    if test == False:
    
        gradient_booster.fit(x_train, y_train)
        y_preds = gradient_booster.predict(x_train)
        
        return y_preds

    if test == True:
        gradient_booster.fit(x_train, y_train)
        y_preds = gradient_booster.predict(x_test)

        return y_preds

#################################################################
############         XG Boosting Classifier       ##############     
    #######       Uses XG Boosting Algorthm       #######
#################################################################

def xgboost_model(x_train, y_train, x_test = 0, y_test = 0, test = False):

    xgb_params = {'max_depth'       : 3,
                  'eta'             : 0.01,
                  'silent'          : 0,
                  'eval_metric'     : 'auc',
                  'subsample'       : 0.8,
                  'colsample_bytree': 0.8,
                  'objective'       : 'binary:logistic'}

    
    xgboost = xgb(params = xgb_params,
                 num_boost_round = 2000,
                 verbose_eval = 50,
                 #early_stopping_rounds = 500,
                 #feval = f1_score_cust,
                 #evals = evals,
                 maximize = True)
    xgboost.fit(x_train, y_train)
    
    
    if test == False:
        y_preds = xgboost.predict(x_train)

        return y_preds

    if test == True:
        y_preds = xgboost.predict(x_test)

        return y_preds
    

#################################################################
#########         LightGMB Boosting Classifier       ###########     
#######       Uses Light Gradient Boosting Algorthm       #######
#################################################################

def lgmboost_model(x, y):
    
    lgmboost = LGBMClassifier(
                learning_rate=0.1,
                max_depth = 5,
                n_estimators=200)

    lgmboost.fit(x, y)
    
    y_preds = lgmboost.predict(x)
    
    return y_preds


#################################################################
#########       HistGradientBoosting Classifier      ###########     
#######    Inspired by Light Gradient Boosting Algorthm    ######
#################################################################

def histgradientboost_model(x_train, y_train, x_test = 0, y_test = 0, test = False):
    
    HGboost = HistGradientBoostingClassifier(
                                            learning_rate=0.1,
                                            max_depth = 5)
   
    HGboost.fit(x_train, y_train)
    
    if test == False:
        y_preds = HGBoost.predict(x_train)
        
        return y_preds
        
    if test == True:
        y_preds = HGBoost.predict(x_test)
    
        return y_preds


##########################################################
#########          Cat Boost Classifier        ###########     
#######      Cat Boost Gradient Boosting Algorthm       ##
##########################################################

def catboost_model(x_train, y_train, x_test = 0, y_test = 0, test = False):
    
    catboost_params = {'loss_function' : 'Logloss',
                        'eval_metric' : 'AUC',
                        'verbose' : 200}
                      
    catboost = CatBoostClassifier(params = catboost_params)

    catboost.fit(x_train, y_train, use_best_model = True)#, plot = True)
    
    if test == False:
        y_preds = catboost.predict(x_train)        
        return y_preds

    if test == True:
        y_preds = catboost.predict(x_test)
        return y_preds

####################################################################
#########         Multinomial Naive Bayes Classifier     ###########     
#######     Uses Naive Bayes as Classification Algorithm     #######
####################################################################

def nb_model(x_train, y_train, x_test = 0, y_test = 0, test = False):
    
    naive_bayes = MultinomialNB()
    
    if test == False:
        naive_bayes.fit(x_train, y_train)
        y_preds = naive_bayes.predict(x_train)

        return y_preds
    
    if test == True:
        naive_bayes.fit(x_train, y_train)
        y_preds = naive_bayes.predict(x_test)

        return y_preds



#########################################################
############         Model Testing      ##############
  ######  Call function to test it's performance  ######
########################################################

######  Naive Bayes Model Train ######
def NB_training():
    
    x_train, y_train, x_test, y_test = get_split_data()
    
    NB_y_preds_train = nb_model(x_train, y_train)
    report = classification_report(y_train, NB_y_preds_train)
    print('Naive Bayes train')
    print(report)

######  Naive Bayes Model Test  ######
def NB_test():
    
    x_train, y_train, x_test, y_test = get_split_data()
    
    NB_y_preds_test = nb_model(x_train, y_train, x_test, y_test, test = True)
    report = classification_report(y_test, NB_y_preds_test)
    print('Naive Bayes test')
    print(report)


######  Gradient Booster Model Train ######

def gb_training():

    x_train, y_train, x_test, y_test = get_split_data()
    gb_y_preds_train = gradient_booster_model(x_train, y_train)
    report = classification_report(y_train, gb_y_preds_train)
    print('SKLearn Gradient Booster train')
    print(report)


######  Gradient Booster Model Test  ######
def gb_test():
    
    x_train, y_train, x_test, y_test = get_split_data()
    
    gb_y_preds_test = gradient_booster_model(x_train, y_train, x_test, y_test, test = True)
    report = classification_report(y_test, gb_y_preds_test)
    print('SKLearn Gradient Booster test')
    print(report)

######  Extreme Gradient Boosting Model Train ######

def xgb_training():

    df = get_df()
    df['language'] = df['language'].map({'Python': 3, 'Other': 2, 'Java' : 0, 'JavaScript' : 1})

    x = df['lemmatized']
    y = df['language']

    cv = CountVectorizer()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 7)

    x_train = cv.fit_transform(x_train)
    x_test = cv.transform(x_test)

    xgb_preds_train = xgboost_model(x_train, y_train)
    report = classification_report(y_train, xgb_preds_train)
    print('Extreme Gradient Boosting training')
    print(report)


######  Extreme Gradient Boosting Model Test  ######
def xgb_test():

    df = get_df()
    df['language'] = df['language'].map({'Python': 3, 'Other': 2, 'Java' : 0, 'JavaScript' : 1})

    x = df['lemmatized']
    y = df['language']

    cv = CountVectorizer()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 7)

    x_train = cv.fit_transform(x_train)
    x_test = cv.transform(x_test)

    xgb_preds_test = xgboost_model(x_train, y_train, x_test, y_test, test = True)
    report = classification_report(y_test, xgb_preds_test)
    print('Extreme Gradient Boosting test')
    print(report)