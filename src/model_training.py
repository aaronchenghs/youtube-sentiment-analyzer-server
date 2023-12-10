import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from sklearn.naive_bayes import MultinomialNB


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
    # Create a pipeline with TfidfVectorizer and SMOTE for handling class imbalance
    pipeline = make_pipeline_imb(
        TfidfVectorizer(),
        SMOTE(random_state=42),
        MultinomialNB()
    )

    # Set up the grid search parameters
    parameters = {
        'tfidfvectorizer__max_df': (0.75, 0.85),
        'tfidfvectorizer__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        # Add other hyperparameters here
    }
    data = pd.read_csv('./youtoxic_english_1000.csv')
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    grid_search = GridSearchCV(pipeline, parameters, cv=5)
    grid_search.fit(train_data['Text'], train_data['IsToxic'])

    # Best model after grid search
    model = grid_search.best_estimator_

    # Evaluate the model
    predictions = model.predict(test_data['Text'])
    accuracy = accuracy_score(test_data['IsToxic'], predictions)
    print(f"{MultinomialNB().__class__.__name__} Accuracy:", accuracy)

    return model
