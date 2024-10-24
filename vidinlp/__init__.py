import spacy
from typing import List, Tuple, Dict, Any
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import gensim
from gensim import corpora
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from collections import Counter, defaultdict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


import re

class VidiNLP:
    def __init__(self, model="en_core_web_sm", lexicon_path='lexicon.txt'):
        self.nlp = spacy.load(model)
        self.sia = SentimentIntensityAnalyzer()
        # Initialize attributes for topic modeling
        self.dictionary = None
        self.lda_model = None
        

        # Initialize TF-IDF vectorizer for keyword extraction
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        
        # Load NRC Emotion Lexicon for emotion detection
        self.emotion_lexicon = self.load_nrc_emotion_lexicon(lexicon_path)
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize the input text and returns a list of tokens."""
        doc = self.nlp(text)
        return [token.text for token in doc]
    
    def lemmatize(self, text: str) -> List[str]:
        """Lemmatize the input text and returns a list of lemmatized words."""
        doc = self.nlp(text)
        return [token.lemma_ for token in doc]
    
    def pos_tag(self, text: str) -> List[Tuple[str, str]]:
        """Returns a list of tuples wit the token and its part of speech tag."""
        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc]
    
    def get_ngrams(self, text: str, n: int, top_n: int = 10) -> List[Tuple[str, int]]:
        """Get top-k n-grams from the input text.
        Args: text: str, n: int, top_k: int = 10
        """
        tokens = self.tokenize(text)
        vectorizer = CountVectorizer(ngram_range=(n, n))
        ngram_matrix = vectorizer.fit_transform([' '.join(tokens)])
        ngrams = vectorizer.get_feature_names_out()
        ngram_counts = ngram_matrix.sum(axis=0).A1
        # Ensure integers are properly formatted and use map to convert to native int
        top_ngrams = sorted(zip(ngrams, map(int, ngram_counts)), key=lambda x: x[1], reverse=True)[:top_n]
        return top_ngrams
    

    def get_tfidf_ngrams(self, text, n=2, top_n=10, filter_stop=True):
        """
        Extract top n-grams from a given text based on TF-IDF scores.
        
        :param text: The input text (string).
        :param n: The value of 'n' for the n-grams (default is 2 for bigrams).
        :param top_n: The number of top n-grams to return based on TF-IDF score.
        :return: A list of tuples with the top n-grams and their respective TF-IDF scores.
        """
        # Initialize the TF-IDF Vectorizer with n-grams
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(n, n))
        if filter_stop:        
            tfidf_vectorizer.stop_words = 'english'
        
        
        # Fit and transform the text
        tfidf_matrix = tfidf_vectorizer.fit_transform([text])
        
        # Get feature names (n-grams) and corresponding scores
        feature_names = tfidf_vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray().flatten()
        
        # Create a dictionary of n-grams and their TF-IDF scores
        ngram_scores = dict(zip(feature_names, scores))
        
        # Sort the n-grams by their scores in descending order
        sorted_ngrams = sorted(ngram_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return the top n n-grams
        return sorted_ngrams[:top_n]


    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
            """Analyze the sentiment of the input text using VADER."""
            return self.sia.polarity_scores(text)
    

    def clean_text(self, text: str, is_stop: bool = False, is_alpha: bool = False, is_punct: bool = False, is_num: bool = False, is_html: bool =False) -> str:
        """Clean and preprocess the input text with optional filters.
        Args: is_stop: bool = False, is_alpha: bool = False, is_punct: bool = False, is_num: bool = False, is_html: bool = False
        """
        
        # Remove HTML tags
        if is_html:
            text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Process the text with the NLP model
        doc = self.nlp(text)
        
        # Clean tokens based on the provided options
        cleaned_tokens = []
        
        for token in doc:
            if is_punct and token.is_punct:
                continue  # Skip punctuation if is_punct is True
            if is_stop and token.is_stop:
                continue  # Skip stopwords if is_stop is True
            if is_alpha and not token.is_alpha:
                continue  # Skip non-alphabetic words if is_alpha is True
            if is_num and token.like_num:
                continue  # Skip numbers if is_num is True
            
            # Add the cleaned token (lowercased)
            cleaned_tokens.append(token.text.lower())
        
        # Return the cleaned text as a single string
        return ' '.join(cleaned_tokens)


    def get_named_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract named entities from the input text and returns a list of tuples with entities and labels."""
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

    def preprocess_for_topic_modeling(self, texts):
        """Preprocess texts for topic modeling."""
        processed_texts = []
        for text in texts:
            doc = self.nlp(text)
            processed_texts.append([token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha])
        return processed_texts

    def train_topic_model(self, texts, num_topics=5, passes=15):
        """Train an LDA topic model on the given texts."""
        processed_texts = self.preprocess_for_topic_modeling(texts)
        self.dictionary = corpora.Dictionary(processed_texts)
        corpus = [self.dictionary.doc2bow(text) for text in processed_texts]
        self.lda_model = gensim.models.LdaModel(corpus=corpus, id2word=self.dictionary, num_topics=num_topics, passes=passes)

    def get_topics(self, num_words=10):
        """Get the topics from the trained LDA model."""
        if self.lda_model is None:
            raise ValueError("Topic model has not been trained. Call train_topic_model first.")
        return self.lda_model.print_topics(num_words=num_words)

    def get_document_topics(self, text):
        """Get the topic distribution for a given document."""
        if self.lda_model is None or self.dictionary is None:
            raise ValueError("Topic model has not been trained. Call train_topic_model first.")
        
        processed_text = self.preprocess_for_topic_modeling([text])[0]
        bow = self.dictionary.doc2bow(processed_text)
        return self.lda_model.get_document_topics(bow)

    def compute_document_similarity(self, doc1, doc2):
        """Compute the similarity between two documents using TF-IDF and cosine similarity."""
        # Preprocess documents
        preprocessed_docs = [self.clean_text(doc) for doc in [doc1, doc2]]
        
        # Create TF-IDF matrix
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(preprocessed_docs)
        
        # Compute cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return similarity

    def find_similar_documents(self, query_doc, document_list, top_n=5):
        """Find the top N most similar documents to the query document."""
        # Preprocess all documents including the query
        preprocessed_docs = [self.clean_text(doc) for doc in [query_doc] + document_list]
        
        # Create TF-IDF matrix
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(preprocessed_docs)
        
        # Compute cosine similarity between query and all other documents
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        # Get indices of top N similar documents
        similar_doc_indices = cosine_similarities.argsort()[:-top_n-1:-1]
        
        # Return list of tuples (document index, similarity score)
        return [(idx, cosine_similarities[idx]) for idx in similar_doc_indices]
    
    def aspect_based_sentiment_analysis(self, text: str) -> Dict[str, Dict[str, float]]:
        """
        Perform aspect-based sentiment analysis on the input text using custom sentiment analysis.

        Args:
        text (str): The input text to analyze.

        Returns:
        Dict[str, Dict[str, Any]]: A dictionary where keys are aspects and values are dictionaries
                                containing sentiment scores, confidence, and the associated text snippet.
        """
        doc = self.nlp(text)
        aspects = defaultdict(list)

        # Extract aspects (nouns) and their associated descriptors
        for token in doc:
            # Check if the token is a noun (aspect)
            if token.pos_ == "NOUN" or token.dep_ == 'compound':
                has_modifier = False  # Track if aspect has any modifiers

                # Check for adjectives, adverbs, or other descriptors linked to the noun
                for child in token.children:
                    if child.dep_ in ["amod", "advmod", "nsubj", "attr", "prep"]:
                        aspects[token.text].append((child.text, token.sent))
                        has_modifier = True

                # If no modifiers are found, associate the whole sentence with the aspect
                if not has_modifier:
                    aspects[token.text].append((None, token.sent))

        # Analyze sentiment for each aspect
        results = {}

        for aspect, modifiers_and_sentences in aspects.items():
            sentiment_scores = []
            confidence_scores = []
            snippets = []

            for modifier, sentence in modifiers_and_sentences:
                if modifier:  # If there's a modifier, analyze the phrase
                    phrase = f"{modifier} {aspect}"
                    sentiment = self.analyze_sentiment(phrase)
                    sentiment_scores.append(sentiment['compound'])
                    confidence_scores.append(abs(sentiment['compound']))  # Confidence based on absolute compound score
                    snippets.append(phrase)
                else:  # If no modifier, analyze the full sentence
                    sentence_text = sentence.text
                    sentiment = self.analyze_sentiment(sentence_text)
                    sentiment_scores.append(sentiment['compound'])
                    confidence_scores.append(abs(sentiment['compound']))  # Confidence based on absolute compound score
                    snippets.append(sentence_text)

            # Average sentiment and confidence for the aspect
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

            results[aspect] = {
                'sentiment': avg_sentiment,
                'confidence': avg_confidence,
                'snippets': snippets
            }

        return results




    def summarize_absa_results(self, absa_results: Dict[str, Dict[str, float]]) -> str:
        """
        Summarize the aspect-based sentiment analysis results into a human-readable format.

        Args:
        absa_results (Dict[str, Dict[str, float]]): The aspect-based sentiment analysis results.

        Returns:
        str: A summary of the results.
        """
        summary = []
        
        for aspect, sentiment_data in absa_results.items():
            # Extract sentiment and confidence scores directly as floats
            sentiment_score = sentiment_data['sentiment']  # This is already a float
            confidence = sentiment_data['confidence']
            
            if sentiment_score > 0.25:
                sentiment_desc = "positive"
            elif sentiment_score < -0.25:
                sentiment_desc = "negative"
            else:
                sentiment_desc = "neutral"
            
            summary.append(f"The aspect '{aspect}' has a {sentiment_desc} sentiment "
                        f"with a confidence of {confidence:.2f}.")

        return "\n".join(summary)

# Additional utility function
@lru_cache(maxsize=1)
def load_spacy_model(model_name: str):
    """Load and cache the spaCy model."""
    return spacy.load(model_name)
