# VidiNLP Library

VidiNLP is a comprehensive Natural Language Processing (NLP) library that combines various text analysis capabilities including sentiment analysis, topic modeling, readability assessment, and more. Built on top of spaCy, scikit-learn, and other powerful NLP tools, VidiNLP provides an easy-to-use interface for advanced text analysis.

For emotion analysis (not sentiment analysis), VidiNLP makes use of the NRC emotion lexicon, created by [Dr Saif M. Mohammad at the National Research Council Canada.](https://www.saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)"

## Table of Contents

- [Installation](#installation)
- [Dependencies](#dependencies)
- [Basic Usage](#basic-usage)
- [Core Features](#core-features)
  - [Text Preprocessing](#text-preprocessing)
  - [Sentiment and Emotion Analysis](#sentiment-and-emotion-analysis)
  - [Keyword Extraction](#keyword-extraction)
  - [Topic Modeling](#topic-modeling)
  - [Document Similarity](#document-similarity)
  - [Readability Analysis](#readability-analysis)
  - [Text Structure Analysis](#text-structure-analysis)
  - [Export Functionality](#export-functionality)

## Installation

```bash
# First install the required dependencies

git clone https://github.com/Vidito/vidinlp.git
cd vidinlp
pip install .


# Download the spaCy model
python -m spacy download en_core_web_sm
```

## Dependencies

- spacy
- scikit-learn
- gensim
- vaderSentiment
- numpy
- pandas

## Basic Usage

```python
from vidinlp import VidiNLP

# Initialize the analyzer
nlp = VidiNLP()
```

## Core Features

### Text Preprocessing

#### Tokenization

```python
# Tokenize text into individual words
tokens = nlp.tokenize("Hello world, how are you?")
print(tokens)
# Output: ['Hello', 'world', ',', 'how', 'are', 'you', '?']
```

#### Lemmatization

```python
# Get base forms of words
lemmas = nlp.lemmatize("I am running and jumping")
print(lemmas)
# Output: ['I', 'be', 'run', 'and', 'jump']
```

#### POS Tagging

```python
# Get part-of-speech tags
pos_tags = nlp.pos_tag("The quick brown fox jumps")
print(pos_tags)
# Output: [('The', 'DET'), ('quick', 'ADJ'), ('brown', 'ADJ'), ('fox', 'NOUN'), ('jumps', 'VERB')]
```

#### Text Cleaning

```python
# Clean text with various filters
cleaned = nlp.clean_text(
    "Hello! This is a test123... <p>with HTML</p>",
    is_stop=True,      # Remove stop words
    is_punct=True,     # Remove punctuation
    is_num=True,       # Remove numbers
    is_html=True       # Remove HTML tags
)
print(cleaned)
# Output: "hello test html"
```

### Sentiment and Emotion Analysis

#### Basic Sentiment Analysis

```python
# Get sentiment scores
sentiment = nlp.analyze_sentiment("This movie was absolutely fantastic!")
print(sentiment)
# Output: {'neg': 0.0, 'neu': 0.227, 'pos': 0.773, 'compound': 0.6369}
```

#### Emotion Analysis

```python
# Get emotion scores
emotions = nlp.analyze_emotions("I am so happy and excited about this!")
print(emotions)
# Output: {'joy': 2, 'anticipation': 1, 'trust': 1, 'surprise': 0, 'fear': 0, 'sadness': 0, 'anger': 0, 'disgust': 0}
```

#### Aspect-Based Sentiment Analysis

```python
# Analyze sentiment for different aspects
absa = nlp.aspect_based_sentiment_analysis(
    "The phone's battery life is excellent but the camera quality is poor."
)
print(nlp.summarize_absa_results(absa))
# Output:
# The aspect 'battery life' has a positive sentiment with a confidence of 0.85.
# The aspect 'camera' has a negative sentiment with a confidence of 0.72.
```

### Keyword Extraction

#### N-gram Analysis

```python
# Get top bigrams
ngrams = nlp.get_ngrams("The quick brown fox jumps over the lazy dog", n=2, top_n=3)
print(ngrams)
# Output: [('quick brown', 1), ('brown fox', 1), ('fox jumps', 1)]
tfidf_ngrams = nlp.get_tfidf_ngrams_corpus(corpus, n=2, top_n=10, filter_stop=False)
# give it a list o texts as corpus
```

#### TF-IDF Keywords

```python
# Extract keywords using TF-IDF
keywords = nlp.extract_keywords("Machine learning is a subset of artificial intelligence", top_k=3)
print(keywords)
# Output: [('machine learning', 0.42), ('artificial intelligence', 0.38), ('subset', 0.20)]
```

### Topic Modeling

```python
# Train topic model
texts = [
    "Machine learning is fascinating",
    "AI and deep learning are revolutionary",
    "Data science uses statistical methods"
]
nlp.train_topic_model(texts, num_topics=2)

# Get topics
topics = nlp.get_topics(num_words=3)
print(topics)
# Output: [(0, 'learning machine deep'), (1, 'data science statistical')]

# Get document topics
doc_topics = nlp.get_document_topics("AI and machine learning")
print(doc_topics)
# Output: [(0, 0.85), (1, 0.15)]
```

### Document Similarity

```python
# Compare two documents
similarity = nlp.compute_document_similarity(
    "Machine learning is fascinating",
    "AI is amazing"
)
print(similarity)
# Output: 0.42

# Find similar documents
docs = ["AI is great", "Machine learning is cool", "Python programming"]
similar = nlp.find_similar_documents("AI and ML", docs, top_n=2)
print(similar)
# Output: [(0, 0.82), (1, 0.65)]
```

### Readability Analysis

```python
# Get readability metrics
readability = nlp.analyze_readability(
    "The quick brown fox jumps over the lazy dog. It was a simple sentence."
)
print(readability)
# Output: {
#     'flesch_reading_ease': 97.0,
#     'gunning_fog_index': 2.8,
#     'dale_chall_score': 5.1,
#     'avg_words_per_sentence': 7.0,
#     'avg_syllables_per_word': 1.2,
#     'complex_word_ratio': 0.0,
#     'lexical_density': 0.571,
#     'type_token_ratio': 0.929,
#     'avg_word_length': 3.93,
#     'named_entity_ratio': 0.0,
#     'verb_noun_ratio': 0.33,
#     'avg_sentence_length_syllables': 8.5
# }
```

### Text Structure Analysis

```python
# Analyze text structure
structure = nlp.analyze_text_structure(
    "This is a sentence. This is another one.\n\nThis is a new paragraph."
)
print(structure)
# Output: {
#     'num_sentences': 3,
#     'avg_sentence_length': 4.33,
#     'num_paragraphs': 2,
#     'avg_paragraph_length': 5.5,
#     'lexical_diversity': 0.76,
#     'complex_sentence_ratio': 0.0
# }
# Analyze text patterns
structure = nlp.detect_linguistic_patterns(
    "The words have been spoken. If they answer, I will talk."
)
print(structure)
# Output: {'passive_voice': ['The words have been spoken],
# 'conditionals': ['If they answer, I will talk.']
# }
```

### Name Entity Recognition

```python
# Identify named entities
ner = nlp.get_named_entities('Norway is a big country!')
print(ner)
# Output: [('Norway', 'GPE')]
```

### Export Functionality

```python
# Export complete analysis in different formats
# JSON format
analysis_json = nlp.export_analysis(text, format='json')

# Pandas DataFrame
analysis_df = nlp.export_analysis(text, format='dataframe')
analysis_df.to_csv('analysis.csv', index=False)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
