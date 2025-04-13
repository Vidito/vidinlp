# setup.py
from setuptools import setup, find_packages

setup(
    name="vidinlp",
    version="1.1.1",
    packages=find_packages(),
    install_requires=[
        "spacy>=3.0.0",
        "scikit-learn>=0.24.0",
        "pandas>=1.0.0",
        "numpy>=1.19.0",
        "vaderSentiment>=3.3.2",
    ],
    package_data={
        "vidinlp": ["data/*.txt"],  # Include all .txt files in the `data` folder
    },
    author="Vahid Niamadpour",
    author_email="contact@pythonology.eu",
    description="A simple NLP library built on top of spaCy, Scikit-Learn, and vadersentiment.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/vidito/vidinlp",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
