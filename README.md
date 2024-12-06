# Fake News Detection with Machine Learning

## Overview

In today’s digital age, distinguishing between real and fake news has become a significant challenge. This project aims to tackle that issue by using machine learning techniques to automatically classify news articles as **real** or **fake** based on their content.

This system uses natural language processing (NLP) to preprocess the data, vectorize the text, and build a machine learning model for accurate classification. It leverages the power of Python libraries such as **scikit-learn**, **nltk**, and **streamlit** to deliver a reliable and interactive solution.

## Features

- **Real-time Fake News Detection**: Classifies news articles as either real or fake with high accuracy.
- **Text Preprocessing**: Removes unwanted characters, stop words, and applies stemming to make the data ready for analysis.
- **TF-IDF Vectorization**: Converts text data into a numerical format that the machine learning model can process effectively.
- **Model Evaluation**: Uses accuracy score to evaluate the model’s performance on unseen data.
- **Interactive Streamlit Web App**: A simple user interface that allows you to test the model with new articles and get real-time predictions.

## How It Works

1. **Data Collection**: The dataset used in this project is a collection of labeled news articles (real vs. fake). The data is downloaded from **Kaggle** and cleaned for use in the model.

2. **Preprocessing**: 
   - Removal of non-alphabetic characters.
   - Lowercasing all words to standardize the text.
   - Tokenization (splitting text into individual words).
   - Removing common stopwords (e.g., 'the', 'and', 'in').
   - Stemming: Reduces words to their root form (e.g., "running" becomes "run").

3. **Machine Learning Model**: 
   - The **Logistic Regression** algorithm is used to build the classification model. This model learns from the training data and predicts whether a news article is real or fake.

4. **Evaluation**: The model is evaluated using the **accuracy score**, ensuring that the classifier performs well on test data.

5. **Streamlit Web Interface**: 
   - The application lets users enter a text snippet or upload an article, and the model will predict whether it's real or fake.

