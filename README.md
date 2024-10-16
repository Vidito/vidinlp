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
# Download spaCy model
# python -m spacy download en_core_web_sm

from vidinlp import VidinNLP

# Initialize the VidinNLP object
nlp = VidinNLP()

# Tokenization
tokens = nlp.tokenize("Hello, world!")

# Lemmatization
lemmas = nlp.lemmatize("The cats are running in the park")

# N-grams
ngrams = nlp.get_ngrams("This is a sample text for n-gram extraction", n=2, top_k=5)

# Sentiment Analysis (pre-trained model)
sentiment = nlp.analyze_sentiment("This is a great library!")

# Text cleaning
cleaned_text = nlp.clean_text("This is some 123 dirty text!!! :)")

# Named Entity Recognition
entities = nlp.get_named_entities("Apple is looking at buying U.K. startup for $1 billion")

#keyword extraction
keywords = nlp.extract_keywords(text, top_k= 10)
```

For more detailed usage instructions and examples, please refer to the documentation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
