# Sentiment analysis with Multinomial Naive Bayes Classifier

This was the first project in Machine Learning course on [Faculty Of Computer Science](https://raf.edu.rs/).

## Problem description

The project included several parts. This repository contains solutions for each assignment. 

- Linear regression - **2a.py** and **2b.py**
- Instance-based models (KNN on Iris dataset) - **3a.py, 3b.py** and **3c.py**
- Multinomial Naive Bayes Classifier (Sentiment analysis of IMDB movie reviews) - **4.py**

A detailed explanation is in [statement.pdf](https://github.com/mmilunovic/ml-homework/blob/master/statement.pdf)

## Dataset

Dataset included 2500 [IMDB movie reviews](https://www.kaggle.com/iarunava/imdb-movie-reviews-dataset), 1250 negative and 1250 positive. 

## Solution

For the first two problems I used [TensorFlow](http://tensorflow.org), and the fourth problem was done in pure python while text processing was done with [nltk](http://www.nltk.org) library. The solution for the fourth problem was fairly simple, I used **BoW histogram** for feature extraction and on top of that straight forward **Multinomial Naive Bayes**. This method produces an accuracy of **90 - 94 %** on the training set and **80 - 87 %** on the test set.

## Prerequisites

- TensorFlow
- SciPy
- nltk


