import spacy
from typing import List, Tuple
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib
import os
from functools import lru_cache
import numpy as np
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter

class VidinNLP:
    def __init__(self, model="en_core_web_sm"):
        self.nlp = spacy.load(model)
        
        # Load pre-trained sentiment model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pipeline_path = os.path.join(current_dir, 'best_sentiment_pipeline_calibrated.joblib')
        self.sentiment_pipeline = joblib.load(pipeline_path)

        # Initialize TF-IDF vectorizer for keyword extraction
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize the input text."""
        doc = self.nlp(text)
        return [token.text for token in doc]
    
    def lemmatize(self, text: str) -> List[str]:
        """Lemmatize the input text."""
        doc = self.nlp(text)
        return [token.lemma_ for token in doc]
    
    def pos_tag(self, text: str) -> List[Tuple[str, str]]:
        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc]
    
    def get_ngrams(self, text: str, n: int, top_k: int = 10) -> List[Tuple[str, int]]:
        """Get top-k n-grams from the input text."""
        tokens = self.tokenize(text)
        vectorizer = CountVectorizer(ngram_range=(n, n))
        ngram_matrix = vectorizer.fit_transform([' '.join(tokens)])
        ngrams = vectorizer.get_feature_names_out()
        ngram_counts = ngram_matrix.sum(axis=0).A1
        top_ngrams = sorted(zip(ngrams, ngram_counts), key=lambda x: x[1], reverse=True)[:top_k]
        return top_ngrams
    
    def analyze_sentiment(self, text: str) -> str:
        """Analyze the sentiment of the input text and return a human-readable result."""
        
        # Check if the sentiment_pipeline is correctly loaded
        if not hasattr(self, 'sentiment_pipeline') or not callable(getattr(self.sentiment_pipeline, 'predict', None)):
            raise ValueError("The sentiment_pipeline is not correctly initialized.")

        # Predict sentiment label
        sentiment = self.sentiment_pipeline.predict([text])[0]
        
        # Get the probability estimates for each class
        probas = self.sentiment_pipeline.predict_proba([text])[0]
        
        # Get the confidence score for the predicted sentiment
        confidence_score = max(probas)  # Highest probability score
        
        # Create a readable output
        result = f"The sentiment is '{sentiment}' with a confidence of {round(confidence_score * 100, 2)}%."
        
        return result
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess the input text."""
        doc = self.nlp(text)
        cleaned_tokens = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
        return ' '.join(cleaned_tokens)

    def get_named_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract named entities from the input text."""
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def extract_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords from the input text using TF-IDF and POS tagging.
        
        Args:
        text (str): The input text to extract keywords from.
        top_k (int): The number of top keywords to return.
        
        Returns:
        List[Tuple[str, float]]: A list of tuples containing keywords and their scores.
        """
        doc = self.nlp(text)
        
        # Preprocess text: lemmatize and remove stopwords, punctuation, and non-alphabetic tokens
        processed_text = ' '.join([token.lemma_.lower() for token in doc 
                                   if not token.is_stop and not token.is_punct and token.is_alpha])
        
        # Calculate TF-IDF scores
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([processed_text])
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = dict(zip(feature_names, tfidf_matrix.toarray()[0]))
        
        # Calculate POS tag scores (prioritize nouns, adjectives, and verbs)
        pos_scores = Counter()
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN']:
                pos_scores[token.lemma_.lower()] += 3
            elif token.pos_ in ['ADJ', 'VERB']:
                pos_scores[token.lemma_.lower()] += 2
            elif token.pos_ == 'ADV':
                pos_scores[token.lemma_.lower()] += 1
        
        # Combine TF-IDF and POS scores
        combined_scores = {word: tfidf_scores.get(word, 0) * (1 + 0.1 * pos_scores.get(word, 0)) 
                           for word in set(tfidf_scores) | set(pos_scores)}
        
        # Sort and return top k keywords
        top_keywords = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return top_keywords

# Additional utility function
@lru_cache(maxsize=1)
def load_spacy_model(model_name: str):
    """Load and cache the spaCy model."""
    return spacy.load(model_name)