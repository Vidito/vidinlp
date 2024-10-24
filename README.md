# VidiNLP

VidiNLP is a simple, modern, and fast NLP library built on top of spaCy. It also uses Gensim for Topic Modelling, and Vader for sentiment analysis. VidiNLP provides easy-to-use functions for common NLP tasks such as

- Tokenization
- Lemmatization
- n-gram extraction
- Sentiment analysis
- Aspect based sentment analysis
- Text cleaning
- Keyword extraction
- Emotion analysis
- Topic modelling
- Document similarity

VidiNLP was built by Dr. Vahid Niamadpour, an applied linguist based in Norway, as a hobby.

For exmotion analysis (not sentiment analusis), VidiNLP makes use of the NRC emotion lexicon, created by [Dr Saif M. Mohammad at the National Research Council Canada.](https://www.saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm) "

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

# Text cleaning: set the arguments you want to be removed from text to True
cleaned_text = nlp.clean_text(text, is_stop = False, is_alpha = False, is_punct = False, is_num = False, is_html = False)

# N-grams. By default the top five bigrams are returned.
ngrams = nlp.get_ngrams(text, n=2, top_n=5)

# N-grams usind TF-IDF. By default the top five bigrams are returned.
tfidf_ngrams = nlp.get_tfidf_ngrams(text, n=2, top_n=5, filter_stop=True)

# Sentiment Analysis (pre-trained model). Example output:
sentiment = nlp.analyze_sentiment(text)
print(sentiment) # {'neg': 0.228, 'neu': 0.345, 'pos': 0.427, 'compound': 0.7777}
print(sentiment['compound']) # 0.7777 you can consider numbers above 0.25 positive, below -0.25 negative, and in between neutral

# Aspect related sentiment analysis
absa = nlp.aspect_based_sentiment_analysis('The phone has a terrible camera.') # returns a dictionary
print(absa)
# {'phone': {'sentiment': -0.4767, 'confidence': 0.4767, 'snippets': ['\nThe phone has a terrible camera.\n']}, 'camera': {'sentiment': -0.4767, 'confidence': 0.4767, 'snippets': ['terrible camera']}}

summary = nlp.summarize_absa_results(absa) # returns a reader friendly format
print(summary)
# The aspect 'phone' has a negative sentiment with a confidence of 0.48.
# The aspect 'camera' has a negative sentiment with a confidence of 0.48.

# Named Entity Recognition. Detects the names of people, cities, countries, or numbers, dates,...
entities = nlp.get_named_entities("Apple is looking at buying U.K. startup for $1 billion")

# Keyword extraction by using TF-IDF and POS tagging
keywords = nlp.extract_keywords(text, top_k= 10)

# Analyze emotions in text using NRC Dictionary. Returns dictionary of emotions and their respective scores.
emotions = nlp.analyze_emotions(text)

# Topic modelling using Gensim
texts = ["Your first document", "Your second document", ...]
nlp.train_topic_model(texts, num_topics=5)
topics = nlp.get_topics()
doc_topics = nlp.get_document_topics("A new document to analyze")


# Document Similarity
# Computes the similarity between two documents using TF-IDF and cosine similarity.
similarity = nlp.compute_document_similarity("First document", "Second document")

# Finds the top N most similar documents to a query document from a list of documents.
similar_docs = nlp.find_similar_documents("Query document", ["Doc1", "Doc2", "Doc3"], top_n=2)

```

For more detailed usage instructions and examples, please refer to the documentation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
