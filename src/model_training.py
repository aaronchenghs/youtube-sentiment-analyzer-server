import os

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.pipeline import make_pipeline as make_pipeline_imb, make_pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB

from constants import trained_model_path


def load_data(file_path, test_size=0.2, random_state=42):
    """
    Loads data from a CSV file, preprocesses it, and splits it into train and test sets.

    Parameters:
    - file_path: str, path to the CSV file.
    - test_size: float, proportion of the dataset to include in the test split.
    - random_state: int, controls the shuffling applied to the data before applying the split.

    Returns:
    - train_data: DataFrame, training data.
    - test_data: DataFrame, test data.
    """

    # Load the dataset
    data = pd.read_csv(file_path)

    # Preprocessing steps (if any), like handling missing values, can be added here

    # Split the dataset into training and testing sets
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)

    return train_data, test_data


def train_model():
    data = pd.read_csv('./training_data.csv')
    label_columns = ['IsToxic', 'IsAbusive', 'IsThreat', 'IsProvocative', 'IsObscene', 'IsHatespeech', 'IsRacist',
                     'IsNationalist', 'IsSexist', 'IsHomophobic', 'IsReligiousHate', 'IsRadicalism']
    X = data['Text']
    y = data[label_columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = make_pipeline(
        TfidfVectorizer(),
        MultiOutputClassifier(MultinomialNB())
    )

    parameters = {
        'tfidfvectorizer__max_df': (0.75, 0.85),
        'tfidfvectorizer__ngram_range': ((1, 1), (1, 2))
    }

    grid_search = GridSearchCV(pipeline, parameters, cv=5)
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

    joblib.dump(model, 'trained_multi_label_model.joblib')
    return model

def get_or_train_model():
    # Check if the model has already been trained and saved
    if os.path.exists(trained_model_path):
        # Load the pre-trained model
        model = joblib.load(trained_model_path)
        print("Loaded pre-trained model.")
    else:
        # Train and save the model
        model = train_model()
        print("Trained and saved new model.")

    return model
