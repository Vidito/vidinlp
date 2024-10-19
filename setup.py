# setup.py
from setuptools import setup, find_packages

setup(
    name="vidinlp",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "spacy",
        "scikit-learn",
        "joblib",
        "pandas",
        "imblearn",
        "numpy",
        "gensim-4.3.3"
    ],
    package_data={
        "vidinlp": ["best_sentiment_pipeline_calibrated.joblib", "vectorizer_svm.joblib"],
    },
    author="Vahid Niamadpour",
    author_email="contact@pythonology.eu",
    description="A simple, modern, and fast NLP library built on top of spaCy",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/vidito/vidinlp",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
