# VidiNLP

VidiNLP is a simple, modern, and fast NLP library built on top of spaCy. It provides easy-to-use functions for common NLP tasks such as tokenization, lemmatization, n-gram extraction, sentiment analysis, text cleaning, and keyword extraction.

## Installation

```
git clone https://github.com/Vidito/vidinlp.git
cd vidinlp
pip install .

```

## Usage

```python
# Download spaCy model like below
# python -m spacy download en_core_web_sm

from vidinlp import VidiNLP

# Initialize the VidiNLP object
nlp = VidiNLP()

# Tokenization
tokens = nlp.tokenize(text)

# Lemmatization
lemmas = nlp.lemmatize(text)

# N-grams. By defaulr the top five bigrams are returned.
ngrams = nlp.get_ngrams(text, n=2, top_k=5)

# Sentiment Analysis (pre-trained model). Example output: "The sentiment is 'Positive' with a confidence of 93%."
sentiment = nlp.analyze_sentiment(text)

# Text cleaning: set the arguments you want to be removed from text to True
cleaned_text = nlp.clean_text(text, is_stop = False, is_alpha = False, is_punct = False, is_num = False, is_html = False)

# Named Entity Recognition. Detects the names of people, cities, countries, or numbers, dates,...
entities = nlp.get_named_entities("Apple is looking at buying U.K. startup for $1 billion")

# Keyword extraction by using TF-IDF and POS tagging
keywords = nlp.extract_keywords(text, top_k= 10)

# Analyze emotions in text using NCR Dictionary. Returns dictionary of emotions and their respective scores.
emotions = nlp.analyze_emotions(text)
```

For more detailed usage instructions and examples, please refer to the documentation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
