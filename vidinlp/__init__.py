import spacy
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib
import os
from functools import lru_cache
from collections import Counter
import re

class VidiNLP:
    def __init__(self, model="en_core_web_sm", lexicon_path='lexicon.txt'):
        self.nlp = spacy.load(model)
        
        # Load pre-trained sentiment model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pipeline_path = os.path.join(current_dir, 'best_sentiment_pipeline_calibrated.joblib')
        self.sentiment_pipeline = joblib.load(pipeline_path)

        # Initialize TF-IDF vectorizer for keyword extraction
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        
        # Load NRC Emotion Lexicon for emotion detection
        self.emotion_lexicon = self.load_nrc_emotion_lexicon(lexicon_path)
        
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
        # Ensure integers are properly formatted and use map to convert to native int
        top_ngrams = sorted(zip(ngrams, map(int, ngram_counts)), key=lambda x: x[1], reverse=True)[:top_k]
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
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces 
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
        List[Tuple[str, float]]: A list of tuples containing keywords and their scores, rounded to 2 decimal points.
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
        
        # Sort and return top k keywords with scores rounded to 2 decimal points
        top_keywords = sorted([(word, round(float(score), 2)) for word, score in combined_scores.items()],
                              key=lambda x: x[1], reverse=True)[:top_k]
        
        return top_keywords
    
    def load_nrc_emotion_lexicon(self, lexicon_path: str) -> Dict[str, Dict[str, int]]:
        """
        Load the NRC Emotion Lexicon into a dictionary.

        Args:
        lexicon_path (str): Path to the NRC Emotion Lexicon file.

        Returns:
        Dict[str, Dict[str, int]]: A dictionary mapping words to emotions and their respective scores (1 or 0).
        """
        lexicon = {}
        with open(lexicon_path, 'r') as file:
            for line in file:
                word, emotion, score = line.strip().split('\t')
                if word not in lexicon:
                    lexicon[word] = {}
                lexicon[word][emotion] = int(score)
        return lexicon

    def analyze_emotions(self, text: str) -> Dict[str, int]:
        """
        Analyze the emotions in the input text using the NRC Emotion Lexicon.

        Args:
        text (str): The input text to analyze emotions from.

        Returns:
        Dict[str, int]: A dictionary of emotions and their respective scores.
        """
        doc = self.nlp(text)
        emotion_scores = Counter()

        for token in doc:
            lemma = token.lemma_.lower()
            if lemma in self.emotion_lexicon:
                for emotion, score in self.emotion_lexicon[lemma].items():
                    emotion_scores[emotion] += score

        return dict(sorted(emotion_scores.items(), key=lambda item: item[1], reverse=True))

# Additional utility function
@lru_cache(maxsize=1)
def load_spacy_model(model_name: str):
    """Load and cache the spaCy model."""
    return spacy.load(model_name)
