# VidinNLP

VidinNLP is a simple, modern, and fast NLP library built on top of spaCy. It provides easy-to-use functions for common NLP tasks such as tokenization, lemmatization, n-gram extraction, sentiment analysis, and text cleaning.

## Installation

```
pip install vidinlp
```

## Usage

```python
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

# Using custom methods
doc = nlp.nlp("This is a sample sentence to demonstrate custom methods.")
print(f"Word count: {doc._.word_count}")

# Named Entity Recognition
entities = get_named_entities("Apple is looking at buying U.K. startup for $1 billion", nlp.nlp)
print(f"Named entities: {entities}")
```

For more detailed usage instructions and examples, please refer to the documentation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
