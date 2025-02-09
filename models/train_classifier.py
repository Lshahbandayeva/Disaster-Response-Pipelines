import sys
import pandas as pd
import numpy as np
import re
import pickle

from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
nltk.download('wordnet') # download for lemmatization
nltk.download('stopwords')


def load_data(database_filepath):
    """This function gets filepath of database file and extract feature, labels, and name of categories.

    Args:
       database_filepath: filepath of database file

    Returns:
       X: feature (message column)
       Y: labels
       category_names: name of categories
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterTable', engine)
    X = df['message']
    Y = df[df.columns[4:]]
    category_names = list(df.columns[4:])
    return X, Y, category_names


def tokenize(text):
    """This function gets the text, tokenize it, and the clean it with lowering, striping, 
    and lemmatizing it. 

    Args:
       text: raw text without any cleaning

    Returns:
       clean_tokens: resulted clean tokens 
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = re.sub(url_regex, 'urlplaceholder', text.lower())

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """This function build a pipeline and train it with set of parameters and train model with 
    best parameters through GridSearchCV.

    Args:
       none

    Returns:
       grid_search: GridSearchCV model
    """
    pipeline = Pipeline([
    ('feature_pipeline', FeatureUnion([

        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer = tokenize)),
            ('tfidf', TfidfTransformer())
        ]))

    ])),

    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__max_depth': [2, 3]
    }

    grid_search = GridSearchCV(pipeline, param_grid = parameters, cv = 3, n_jobs = -1)
    return grid_search
    

def evaluate_model(model, X_test, Y_test, category_names):
    """This function evaluate the model with calculating accuracy score and classification metric 
    (precision, recall, f1 score).

    Args:
       model: model which is needs to be evaluated
       X_test: test features
       Y_test: test labels
       category_names: name of categories

    Returns:
       none
    """
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred)
    i = 0
    for col in Y_test.columns:
        print("Accuracy score: ", accuracy_score(Y_test[col], y_pred[i]))
        print(classification_report(Y_test[col], y_pred[i]))
        i += 1

def save_model(model, model_filepath):
    """This function gets model and its filepath and export it to the pickle file.

    Args:
       model: trained and evaluated model
       model_filepath: filepath where model needs to be exported

    Returns:
       none
    """
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()