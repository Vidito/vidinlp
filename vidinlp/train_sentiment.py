import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
import joblib
import logging

logging.basicConfig(level=logging.INFO)

def train_and_save_sentiment_model():
    file_path = 'sentiment1.csv'

    # Try to read the CSV file with different encodings
    encodings = ['utf-8', 'iso-8859-1', 'cp1252']
    data = None
    
    for encoding in encodings:
        try:
            data = pd.read_csv(file_path, encoding=encoding)
            data = data.dropna()  # Drop rows with missing values
            logging.info(f"Successfully read the CSV file with {encoding} encoding.")
            break
        except (UnicodeDecodeError, FileNotFoundError):
            continue

    if data is None:
        raise ValueError("Unable to read the CSV file. Please check the file path and encoding.")
    
    X = data['text']
    y = data['sentiment']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a TfidfVectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    X_train_vectorized = vectorizer.fit_transform(X_train)

    # Apply SMOTE to the vectorized training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vectorized, y_train)

    # Create a pipeline with CalibratedClassifierCV
    pipeline = Pipeline([
        ('svm', CalibratedClassifierCV(LinearSVC(class_weight='balanced', random_state=42)))
    ])

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'svm__estimator__C': [0.01, 0.1, 1, 10],
        'svm__estimator__max_iter': [1000, 2000],
        'svm__estimator__tol': [1e-3, 1e-4]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search.fit(X_train_resampled, y_train_resampled)

    # Get the best model
    best_model = grid_search.best_estimator_
    logging.info(f"Best parameters: {grid_search.best_params_}")

    # Create a new pipeline that includes the vectorizer and the best model
    final_pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('model', best_model)
    ])

    # Save the entire pipeline
    joblib.dump(final_pipeline, 'best_sentiment_pipeline_calibrated.joblib')
    logging.info("Model pipeline saved successfully.")

    # Evaluate on the test set
    y_pred = final_pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    return final_pipeline

if __name__ == "__main__":
    trained_model = train_and_save_sentiment_model()