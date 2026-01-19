import os
from pathlib import Path
import spacy
from spacy.tokens import Doc
from typing import List, Tuple, Dict, Any, Optional, Union
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from collections import Counter, defaultdict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import pandas as pd
import numpy as np

# Utility function placed at top level for caching effectiveness
@lru_cache(maxsize=1)
def load_spacy_model(model_name: str):
    """Load and cache the spaCy model."""
    try:
        return spacy.load(model_name)
    except OSError:
        raise OSError(
            f"SpaCy model '{model_name}' not found. Please install it using: python -m spacy download {model_name}"
        )

class VidiNLP:
    def __init__(
        self,
        model="en_core_web_sm",
        lexicon_path: Optional[str] = None,
        easy_word_list: Optional[str] = None,
    ):
        """
        Initialize VidiNLP with lazy loading and optimized file handling.
        """
        self.model_name = model
        self._nlp = None  # Lazy load placeholder
        
        self.sia = SentimentIntensityAnalyzer()
        self.dictionary = None
        self.lda_model = None
        self.vectorizer = None
        self.classifier = None

        # Set default paths relative to the current file's location
        current_dir = Path(__file__).parent
        default_lexicon = current_dir / "data" / "lexicon.txt"
        default_word_list = current_dir / "data" / "chall_word_list.txt"

        # Load resources
        self.easy_words = self._load_resource(
            easy_word_list if easy_word_list else default_word_list,
            self.load_easy_word_list,
            default=[]
        )
        
        self.emotion_lexicon = self._load_resource(
            lexicon_path if lexicon_path else default_lexicon,
            self.load_nrc_emotion_lexicon,
            default={}
        )

        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2)
        )

    @property
    def nlp(self):
        """Lazy loader ensures model is only loaded when needed."""
        if self._nlp is None:
            self._nlp = load_spacy_model(self.model_name)
        return self._nlp

    def _ensure_doc(self, text_or_doc: Union[str, Doc]) -> Doc:
        """
        Optimization Helper: returns a Doc object. 
        If input is already a Doc, returns it (preventing double processing).
        If input is str, processes it.
        """
        if isinstance(text_or_doc, Doc):
            return text_or_doc
        if not isinstance(text_or_doc, str):
             # Handle edge case where input might be non-string/non-doc
            return self.nlp(str(text_or_doc))
        return self.nlp(text_or_doc)

    @staticmethod
    def _load_resource(path, loader_func, default):
        """Generic safe resource loader."""
        try:
            return loader_func(path)
        except (FileNotFoundError, Exception) as e:
            print(f"Warning: Resource not loaded from {path}. Using default. Error: {e}")
            return default

    @staticmethod
    def load_easy_word_list(file_path: Union[str, Path]) -> List[str]:
        """Load the Dale-Chall easy word list."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "r", encoding="utf-8") as file:
            return [line.strip().lower() for line in file if line.strip()]

    @staticmethod
    def load_nrc_emotion_lexicon(lexicon_path: Union[str, Path]) -> Dict[str, Dict[str, int]]:
        """Load the NRC Emotion Lexicon."""
        path = Path(lexicon_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
            
        lexicon = {}
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    parts = line.strip().split("\t")
                    if len(parts) != 3: continue
                    word, emotion, score = parts
                    if word not in lexicon:
                        lexicon[word] = {}
                    lexicon[word][emotion] = int(score)
                except ValueError:
                    continue
        return lexicon

    # --- Core NLP Operations ---

    def tokenize(self, text: Union[str, Doc]) -> List[str]:
        doc = self._ensure_doc(text)
        return [token.text for token in doc]

    def lemmatize(self, text: Union[str, Doc]) -> List[str]:
        doc = self._ensure_doc(text)
        return [token.lemma_ for token in doc]

    def pos_tag(self, text: Union[str, Doc]) -> List[Tuple[str, str]]:
        doc = self._ensure_doc(text)
        return [(token.text, token.pos_) for token in doc]

    def get_named_entities(self, text: Union[str, Doc]) -> List[Tuple[str, str]]:
        doc = self._ensure_doc(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def get_ngrams(
        self,
        text: str,
        n: int,
        top_n: int = 10,
        lowercase: bool = True,
        ignore_punct: bool = True,
    ) -> List[Tuple[str, int]]:
        """Fast n-gram extraction using regex (bypasses spaCy pipeline for speed)."""
        if lowercase:
            text = text.lower()
        
        # Regex is significantly faster than spaCy for simple tokenization
        tokens = re.findall(r"\w+(?:'\w+)?", text) if ignore_punct else text.split()
        
        if len(tokens) < n:
            return []

        ngrams = zip(*[tokens[i:] for i in range(n)])
        return Counter(" ".join(g) for g in ngrams).most_common(top_n)

    def get_tfidf_ngrams_corpus(self, corpus, n=2, top_n=10, filter_stop=False):
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(n, n))
        if filter_stop:
            tfidf_vectorizer.set_params(stop_words="english")
        
        try:
            tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
        except ValueError: # Handle empty vocabulary case
            return []

        feature_names = tfidf_vectorizer.get_feature_names_out()
        scores = tfidf_matrix[-1].toarray().flatten()
        
        # Optimize sort: skip creating dict if just needing top_n
        indices = np.argsort(scores)[::-1][:top_n]
        return [(feature_names[i], scores[i]) for i in indices]

    # --- Analysis & Sentiment ---

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """VADER analysis (Does not use SpaCy)."""
        return self.sia.polarity_scores(text)

    def clean_text(
        self,
        text: str,
        remove_stop_words: bool = False,
        remove_none_alpha: bool = False,
        remove_punctuations: bool = False,
        remove_numbers: bool = False,
        remove_html: bool = False,
        remove_urls: bool = False,
        remove_emojis: bool = False,
    ) -> str:
        if not text or not text.strip():
            return ""

        # Optimization: Perform regex cleaning BEFORE NLP pipeline
        # This reduces the token count SpaCy needs to process
        if remove_html:
            text = re.sub(r"<[^>]+>", "", text)
        if remove_urls:
            text = re.sub(r"https?://\S+|www\.\S+", "", text)
        if remove_emojis:
            text = re.sub(r"[^\x00-\x7F]+", "", text) # Simple non-ascii removal for speed
        
        text = re.sub(r"\s+", " ", text).strip()

        # If no linguistic filtering is needed, return early to avoid loading model
        if not any([remove_stop_words, remove_none_alpha, remove_punctuations, remove_numbers]):
            return text.lower()

        # Only invoke NLP if linguistic features are required
        doc = self.nlp(text)
        cleaned_tokens = []

        for token in doc:
            if remove_punctuations and token.is_punct: continue
            if remove_stop_words and token.is_stop: continue
            if remove_none_alpha and not token.is_alpha: continue
            if remove_numbers and token.like_num: continue
            
            cleaned_tokens.append(token.text.lower())

        return " ".join(cleaned_tokens)

    def extract_keywords(self, text: Union[str, Doc], top_n: int = 10) -> List[Tuple[str, float]]:
        doc = self._ensure_doc(text)

        # Extract tokens for TF-IDF
        valid_tokens = [
            token.lemma_.lower() 
            for token in doc 
            if not token.is_stop and not token.is_punct and token.is_alpha
        ]
        
        if not valid_tokens:
            return []

        processed_text = " ".join(valid_tokens)

        # Fit single document
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([processed_text])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            tfidf_scores = dict(zip(feature_names, scores))
        except ValueError:
            return []

        # Calculate scores in a single pass
        combined_scores = {}
        # Pre-weighting POS tags
        pos_weights = {"NOUN": 3, "PROPN": 3, "ADJ": 2, "VERB": 2, "ADV": 1}
        
        for token in doc:
            lemma = token.lemma_.lower()
            if lemma in tfidf_scores:
                weight = pos_weights.get(token.pos_, 0)
                # Formula: TFIDF * (1 + 0.1 * Weight)
                current_score = tfidf_scores[lemma] * (1 + 0.1 * weight)
                # Keep max score if word appears multiple times with diff POS
                combined_scores[lemma] = max(combined_scores.get(lemma, 0), current_score)

        return sorted(
            [(k, round(v, 2)) for k, v in combined_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

    def analyze_emotions(self, text: Union[str, Doc]) -> Dict[str, int]:
        doc = self._ensure_doc(text)
        emotion_scores = Counter()

        for token in doc:
            lemma = token.lemma_.lower()
            if lemma in self.emotion_lexicon:
                for emotion, score in self.emotion_lexicon[lemma].items():
                    emotion_scores[emotion] += score

        return dict(emotion_scores.most_common())

    def topic_modelling(
        self,
        texts: List[str],
        num_topics: int = 5,
        min_df: int = 2,
        max_df: float = 0.95,
        min_word_length: int = 3,
    ) -> List[Dict[str, float]]:
        """Optimization: Uses nlp.pipe for batch processing."""
        processed_texts = []
        
        # nlp.pipe is much faster for lists of text
        # disable NER and Parser as they aren't needed for lemma extraction
        for doc in self.nlp.pipe(texts, disable=["ner", "parser"]):
            tokens = [
                token.lemma_.lower() 
                for token in doc 
                if not token.is_stop and token.is_alpha and len(token.lemma_) >= min_word_length
            ]
            processed_texts.append(" ".join(tokens))

        if not processed_texts:
            return []

        vectorizer = CountVectorizer(stop_words="english", min_df=min_df, max_df=max_df)
        try:
            doc_term_matrix = vectorizer.fit_transform(processed_texts)
        except ValueError:
            return []

        lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            learning_method="online",
            n_jobs=-1 # Use all CPU cores
        )
        lda.fit(doc_term_matrix)

        feature_names = vectorizer.get_feature_names_out()
        topics = []

        for topic_idx, topic in enumerate(lda.components_):
            top_features_ind = topic.argsort()[:-11:-1]
            top_features = [
                {"keyword": feature_names[i], "weight": float(topic[i])}
                for i in top_features_ind
            ]
            topics.append({f"Topic_{topic_idx+1}": top_features})

        return topics

    def compute_document_similarity(self, doc1, doc2):
        # Using simple clean locally to avoid circular dependencies or heavy lifting
        docs = [self.clean_text(d) for d in [doc1, doc2]]
        tfidf = self.tfidf_vectorizer.fit_transform(docs)
        return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    def find_similar_documents(self, query_doc, document_list, top_n=5):
        clean_docs = [self.clean_text(d) for d in [query_doc] + document_list]
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(clean_docs)
        
        cosine_similarities = cosine_similarity(
            tfidf_matrix[0:1], tfidf_matrix[1:]
        ).flatten()

        similar_doc_indices = cosine_similarities.argsort()[::-1][:top_n]
        return [(idx, cosine_similarities[idx]) for idx in similar_doc_indices]

    def aspect_based_sentiment_analysis(self, text: Union[str, Doc]) -> Dict[str, Dict[str, Any]]:
        doc = self._ensure_doc(text)
        aspects = defaultdict(list)

        for token in doc:
            if token.pos_ == "NOUN" or token.dep_ == "compound":
                has_modifier = False
                for child in token.children:
                    if child.dep_ in ["amod", "advmod", "nsubj", "attr", "prep"]:
                        aspects[token.text].append((child.text, token.sent))
                        has_modifier = True
                if not has_modifier:
                    aspects[token.text].append((None, token.sent))

        results = {}
        for aspect, items in aspects.items():
            sentiments = []
            snippets = []
            
            for modifier, sentence in items:
                # Use raw text from span/sentence
                phrase = f"{modifier} {aspect}" if modifier else sentence.text
                snippets.append(phrase)
                
                # Analyze
                score = self.analyze_sentiment(phrase)["compound"]
                sentiments.append(score)

            if not sentiments: continue

            avg_sent = sum(sentiments) / len(sentiments)
            avg_conf = sum(abs(s) for s in sentiments) / len(sentiments)

            results[aspect] = {
                "sentiment": avg_sent,
                "confidence": avg_conf,
                "snippets": snippets,
            }

        return results

    def summarize_absa_results(self, absa_results: Dict) -> str:
        summary = []
        for aspect, data in absa_results.items():
            score = data["sentiment"]
            desc = "positive" if score > 0.25 else "negative" if score < -0.25 else "neutral"
            summary.append(
                f"The aspect '{aspect}' has a {desc} sentiment with confidence {data['confidence']:.2f}."
            )
        return "\n".join(summary)

    def detect_linguistic_patterns(self, text: Union[str, Doc]) -> Dict[str, Any]:
        doc = self._ensure_doc(text)
        
        # Initialize containers
        patterns = {
            "voice": {"passive": [], "active": []},
            "sentence_structure": defaultdict(list),
            "modality": defaultdict(list),
            "cohesion_devices": defaultdict(list),
        }

        # Sets for O(1) lookups
        modal_verbs = {"can", "could", "may", "might", "shall", "should", "will", "would", "must"}
        temporal = {"when", "while", "before", "after", "since", "until", "then", "later"}
        causal = {"because", "since", "therefore", "thus", "hence"}
        contrastive = {"but", "however", "although", "yet", "still", "whereas"}

        for sent in doc.sents:
            sent_text = sent.text.strip()
            tokens_lower = [t.text.lower() for t in sent]
            
            # Voice Detection
            has_passive = any(t.dep_ == "nsubjpass" for t in sent)
            if has_passive:
                # Find subject and verb for passive
                subj = next((t.text for t in sent if t.dep_ == "nsubjpass"), "")
                patterns["voice"]["passive"].append({"text": sent_text, "subject": subj})
            elif any(t.dep_ == "nsubj" for t in sent):
                 # Simple active check
                patterns["voice"]["active"].append({"text": sent_text})

            # Sentence Structure
            clauses = [t for t in sent if t.dep_ == "mark"]
            coords = [t for t in sent if t.dep_ == "cc"]
            
            if not clauses and not coords:
                patterns["sentence_structure"]["simple"].append(sent_text)
            elif clauses and not coords:
                patterns["sentence_structure"]["complex"].append(sent_text)
            elif not clauses and coords:
                patterns["sentence_structure"]["compound"].append(sent_text)
            else:
                patterns["sentence_structure"]["compound_complex"].append(sent_text)

            # Modality & Cohesion (Set intersection is faster than loops)
            sent_words = set(tokens_lower)
            
            if sent_words & {"if", "unless", "whether"}:
                patterns["modality"]["conditionals"].append(sent_text)
            if sent_words & modal_verbs:
                patterns["modality"]["hypotheticals"].append(sent_text)
            
            # Imperative check (verb starts sentence)
            if sent[0].pos_ == "VERB" and sent[0].tag_ == "VB":
                patterns["modality"]["imperatives"].append(sent_text)

            if sent_words & temporal: patterns["cohesion_devices"]["temporal_markers"].append(sent_text)
            if sent_words & causal: patterns["cohesion_devices"]["causal_markers"].append(sent_text)
            if sent_words & contrastive: patterns["cohesion_devices"]["contrastive_markers"].append(sent_text)

        patterns["statistics"] = {
            k: {sub_k: len(v) for sub_k, v in val.items()} 
            for k, val in patterns.items() if k != "statistics"
        }
        return patterns

    def analyze_text_structure(self, text: Union[str, Doc]) -> Dict[str, Any]:
        doc = self._ensure_doc(text)
        sentences = list(doc.sents)
        
        # Stats
        sentence_lengths = [len(sent) for sent in sentences]
        words = [t.text.lower() for t in doc if t.is_alpha]
        pos_counts = Counter(t.pos_ for t in doc)
        
        noun_count = pos_counts["NOUN"]
        verb_count = pos_counts["VERB"]
        adj_count = pos_counts["ADJ"]

        return {
            "num_sentences": len(sentences),
            "avg_sentence_length": np.mean(sentence_lengths) if sentence_lengths else 0,
            "lexical_diversity": len(set(words)) / len(words) if words else 0,
            "pos_distribution": dict(pos_counts),
            "noun_verb_ratio": round(noun_count / verb_count, 2) if verb_count else 0,
            "noun_adj_ratio": round(noun_count / adj_count, 2) if adj_count else 0,
        }

    @staticmethod
    @lru_cache(maxsize=10000)
    def _count_syllables(word: str) -> int:
        """Cached syllable counting for speed."""
        word = word.lower()
        if len(word) <= 3: return 1
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels: count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"): count -= 1
        return max(1, count)

    def analyze_readability(self, text: Union[str, Doc]) -> Dict[str, float]:
        doc = self._ensure_doc(text)
        
        words = [t.text.lower() for t in doc if not t.is_punct]
        word_count = len(words)
        sent_count = len(list(doc.sents))
        
        if sent_count == 0 or word_count == 0:
            return {"error": "Insufficient text"}

        syllable_count = sum(self._count_syllables(w) for w in words)
        
        avg_words_per_sent = word_count / sent_count
        avg_syllables_per_word = syllable_count / word_count
        
        # Optimization: convert list to set for O(1) lookup in difficult word check
        easy_words_set = set(self.easy_words)
        difficult_words = sum(1 for w in words if w not in easy_words_set)
        
        # Scores
        flesch = 206.835 - 1.015 * avg_words_per_sent - 84.6 * avg_syllables_per_word
        
        diff_ratio = difficult_words / word_count
        dale_chall = 0.1579 * (diff_ratio * 100) + 0.0496 * avg_words_per_sent
        if diff_ratio > 0.05: dale_chall += 3.6365

        complex_words = sum(1 for w in words if self._count_syllables(w) >= 3)
        fog = 0.4 * (avg_words_per_sent + 100 * (complex_words / word_count))

        return {
            "flesch_reading_ease": round(flesch, 2),
            "gunning_fog_index": round(fog, 2),
            "dale_chall_score": round(dale_chall, 2),
            "avg_words_per_sentence": round(avg_words_per_sent, 2),
            "complex_word_ratio": round(complex_words / word_count, 3),
        }

    def export_analysis(self, text: str, format: str = "json") -> Union[str, Dict, pd.DataFrame]:
        """
        Export comprehensive text analysis. 
        MAJOR OPTIMIZATION: Runs SpaCy pipeline ONCE and passes Doc to sub-functions.
        """
        # 1. Run Pipeline Once
        doc = self.nlp(text)
        
        # 2. Distribute Pre-computed Doc
        analysis = {
            "basic_stats": {
                "word_count": len(doc),
                "sentence_count": len(list(doc.sents)),
            },
            "sentiment": self.analyze_sentiment(text), # VADER works on string
            "emotions": self.analyze_emotions(doc),
            "keywords": {
                f"keyword_{i+1}": {"text": kw[0], "score": kw[1]}
                for i, kw in enumerate(self.extract_keywords(doc))
            },
            "readability": self.analyze_readability(doc),
            "linguistic_patterns": {
                k: v["statistics"] if "statistics" in v else len(v)
                for k, v in self.detect_linguistic_patterns(doc).items()
            },
            "named_entities": {
                f"entity_{i+1}": {"text": ent[0], "label": ent[1]}
                for i, ent in enumerate(self.get_named_entities(doc))
            },
        }

        if format == "dataframe":
            flattened = []
            for category, values in analysis.items():
                if isinstance(values, dict):
                    for k, v in values.items():
                        if isinstance(v, dict):
                            for sk, sv in v.items():
                                flattened.append({"category": category, "key": f"{k}_{sk}", "value": sv})
                        else:
                            flattened.append({"category": category, "key": k, "value": v})
                else:
                    flattened.append({"category": category, "key": "", "value": values})
            return pd.DataFrame(flattened)

        return analysis

    def train_text_classifier(self, csv_path, text_column, label_column, split_ratio=0.8):
        try:
            data = pd.read_csv(csv_path)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")

        if text_column not in data.columns or label_column not in data.columns:
            raise ValueError(f"Columns not found in dataset.")

        # Clean text before vectorization for better accuracy
        # Note: clean_text is fast now (regex based)
        X = data[text_column].fillna("").astype(str)
        y = data[label_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1 - split_ratio, random_state=42
        )

        self.vectorizer = TfidfVectorizer()
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        self.classifier = MultinomialNB()
        self.classifier.fit(X_train_tfidf, y_train)

        y_pred = self.classifier.predict(X_test_tfidf)
        print("Model Accuracy:", accuracy_score(y_test, y_pred))
        print("Model trained successfully!")

    def predict_text(self, text):
        if not self.classifier or not self.vectorizer:
            raise ValueError("Model not trained.")
        text_tfidf = self.vectorizer.transform([text])
        return self.classifier.predict(text_tfidf)[0]
