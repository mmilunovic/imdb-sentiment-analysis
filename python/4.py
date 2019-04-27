import os
from nltk.tokenize import regexp_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist
import re
import random
import math
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import operator

# Read all review files and store them in file_content list
def read_folder(path_to_folder):
    files = os.listdir(path_to_folder)
    file_content = []
    for i in range(samples):
        f = open(path_to_folder+'\\'+files[i], 'r', encoding="utf8")
        file_content.append(f.read())
        
    return file_content

# Removes html tags from data
def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)

# Calculates accuracy of a model
# For every label check if corresponding prediction from
# predictions list is the same
# Sum all and divide with total number of predictions
def calc_acc(labels, predictions):
        nb_correct = 0
        nb_total = len(labels)
        for i in range(nb_total):
            if labels[i] == predictions[i]:
                nb_correct += 1
        return nb_correct / nb_total
    
# Calculate and print confusion matrix for given actual and predicted values
def print_confusio_matrix(actual, predictions, labels):
    cm = confusion_matrix(actual, predictions)
    df = pd.DataFrame()
    for i, row_label in enumerate(labels):
        rowdata={}
        for j, col_label in enumerate(labels): 
            rowdata[col_label]=cm[i,j]
        df = df.append(pd.DataFrame.from_dict({row_label:rowdata}, orient='index'))
    print(df)  
    
# Contains all operations related to feature extraction and
# data split
class Dataset:
    def __init__(self, path, nb_words, test_ratio, y):
        
        self.nb_words = nb_words
        self.corpus = []
        self.clean_corpus = []
        self.vocab_set = set()
        
        # Loads both neg and pos reviews to corpus
        self.neg_corpus = read_folder(path+"\\neg")
        self.pos_corpus = read_folder(path+"\\pos")
        
        print('Cleaning the corpus...')
        self.clean_neg = self.prepare_corpus(self.neg_corpus)
        self.clean_pos = self.prepare_corpus(self.pos_corpus)
        
        self.clean_corpus = self.clean_neg + self.clean_pos
        
        for doc in self.clean_corpus:
            for word in doc:
                self.vocab_set.add(word)
        self.vocab = list(self.vocab_set)  
               
        self.neg_counts = dict()
        for word in self.vocab:
          self.neg_counts[word] = 0
          for doc in self.clean_neg:
            if word in doc:
              self.neg_counts[word] += 1
        
        self.pos_counts = dict()
        for word in self.vocab:
          self.pos_counts[word] = 0
          for doc in self.clean_pos:
            if word in doc:
              self.pos_counts[word] += 1
        
        # Create BoW feature vector from clean_corpus
        x = self.create_bow(self.clean_corpus)
        
        # Shuffles the data (all negative reviews are loaded first)
        combined = list(zip(x, y))
        random.shuffle(combined)
        x[:], y[:] = zip(*combined)
        
        # Split the data into train and test 
        # Test size is determined by test_ratio
        nb_total = len(x)
        nb_test = int(test_ratio * nb_total)
        nb_train = nb_total - nb_test
 
        self.train = {
            'x': x[:nb_train], 
            'y': y[:nb_train]
        }
        self.test = {
            'x': x[nb_train:],
            'y': y[nb_train:]
        }
        
    # Removes stopwords, html tags and punctuation
    # Stemming with PorterStemmer
    def prepare_corpus(self, corpus):
        clean_corpus = []
        stop_punc = set(stopwords.words('english')).union(set(punctuation))
        porter = PorterStemmer()
        for doc in corpus:
            words = regexp_tokenize(striphtml(doc), "[A-Za-z']+")
            words_lower = [w.lower() for w in words]
            words_filtered = [w for w in words_lower if w not in stop_punc]
            words_stemmed = [porter.stem(w) for w in words_filtered]
            clean_corpus.append(words_stemmed)
        return clean_corpus

    # Creates BoW histogram
    # Counts number of occurrences of 10000 most used words in all reviews
    # Every review is represented as histogram of most used words
    def create_bow(self, clean_corpus):
        freq = FreqDist([w for rew in clean_corpus for w in rew])
        best_words, _ = zip(*freq.most_common(nb_words)) # Most used nb_words(10000) words
        x = []
        for rew in clean_corpus:
            bow = dict()
            for i in range(nb_words):
                cnt = rew.count(best_words[i])
                if cnt > 0:
                    bow[i] = cnt
            x.append(bow)
        return x
    
    # Calculates lr metric as proposed in statement, if top5 argument is True
    # it will calculate and print 5 most used words in pos/neg reviews
    def lr_metric(self, word, top5):
        num_neg = 0
        num_pos = 0
        
        if top5:
            freq = FreqDist([w for rew in self.clean_neg for w in rew])
            best_neg, _ = zip(*freq.most_common(5))
            freq = FreqDist([w for rew in self.clean_pos for w in rew])
            best_pos, _ = zip(*freq.most_common(5))
            print("Top 5 words used in negative reviews:")
            print(best_neg)
            print("Top 5 words used in positive reviews:")
            print(best_pos)
            
        num_neg = self.neg_counts.get(word)
        num_pos = self.pos_counts.get(word)
        
        if num_pos >= 10 and num_neg >= 10:
            return float(num_pos/num_neg)
        else:
            return -1 
    
class MultinomialNaiveBayes:
    """
    A Naive Bayes classifier that uses a categorical distribution to 
    estimate P(xi|C) likelihoods. Feature vector (x) is treated as
    a histogram of event frequencies so we have: 
        P(x|C)=product(P(xi|C))=product(pi^xi), 
    where pi are event probabilities from our categorical distribution.
    
    Therefore: log(P(C|x)) ~ log(P(C)) + sum(xi*log(pi))
    """

    def __init__(self, nb_classes, nb_words, pseudocount):
        self.nb_classes = nb_classes
        self.nb_words = nb_words
        self.pseudocount = pseudocount  # additive smoothing parameter
    
    def fit(self, data):
        x, y = data['x'], data['y']
        nb_examples = len(y)

        # Calculate class priors
        self.priors = np.bincount(y) / nb_examples

        # Calculate number of occurrances of each word in each class
        occs = np.zeros((self.nb_classes, self.nb_words))
        for i in range(nb_examples):
            label = y[i]
            for w, cnt in x[i].items():
                occs[label][w] += cnt
        
        # Calculate event likelihoods for each class
        self.likelihoods = np.zeros((self.nb_classes, self.nb_words))
        for c in range(self.nb_classes):
            for w in range(self.nb_words):
                num = occs[c][w] + self.pseudocount
                down = sum(occs[c]) + self.nb_words*self.pseudocount
                self.likelihoods[c][w] = num / down
            
    def predict(self, bow):
        nb_examples = len(bow)
        probs = []
        for i in range(nb_examples):
            # Calculate log probabilities for each class
            log_probs = []
            for c in range(self.nb_classes):
                log_prob = math.log(self.priors[c])
                for w, cnt in bow[i].items():
                    log_prob += cnt * math.log(self.likelihoods[c][w])
                log_probs.append(log_prob)

            # Max log probability gives the prediction
            pred = np.argmax(log_probs)
            probs.append(pred)
        return probs


path_to_folder = "D:\\RAF\\6. Semestar\\Mašinsko učenje\\ml-homewok\\data\\imdb"
samples = 1250 # Use all data
nb_words = 10000 # As said in statement
nb_classes = 2 # Positive and Negative
pseudocount = 2 # Looks like it gives best solution 

# Make labels for all reviews, first 1250 reviews are negative, and other 1250 are positive
data_Y = []
for i in range(2*samples):
    if i < samples:
        data_Y.append(0)
    else:
        data_Y.append(1)

# Loads dataset from path_to_folder
dataset = Dataset(path_to_folder, nb_words, 0.2, data_Y)

# Train the model with train data (80%)))
print("Tranning...")
model = MultinomialNaiveBayes(nb_classes, nb_words, pseudocount)
model.fit(dataset.train)

preds_train = model.predict(dataset.train['x'])
acc_train = calc_acc(dataset.train['y'], preds_train)
print("Train set accuracy: {:.2f}%".format(round(acc_train*100)))

# Test the model on remaingin 20% of reviews
predictions = model.predict(dataset.test['x'])
acc_test = calc_acc(dataset.test['y'], predictions)
print("Test set accuracy: {:.2f}%".format(round(acc_test*100)))

# Calculate and print confusion matrix
print("Confusion matrix: ")
print_confusio_matrix(dataset.test['y'], predictions, ["Positive","Negative"])

# LR metric calculations
print("LR metric:")
lr_list = dict()
for word in dataset.vocab:
    lr = dataset.lr_metric(word, False)
    if lr != -1:
        lr_list[word] = lr
    
sorted_lr = sorted(lr_list.items(), key=operator.itemgetter(1))
print("5 words with largest lr metric")
print(sorted_lr[-5:])
print("5 words with smallest lr metric")
print("{}".format(sorted_lr[:5]))



'''
Najčešće reči u pozitivnim i negativnim kritikama su iste, uopštene reči vezane za filmove. 
Movie, watch, film, lot, like...like može da bude samo a može i not da ima :) 

LR metrika već daje bolji uvid. Reči sa najvećmo vrednošću su one iz pozitivnih kritika. 
Perfection, excel, delight...
Dok su one sa najmanjom vrednošću iz loših kritika.
Worst, stupid, crap...

Korišćenjem ove metrike a ne čiste frekvencije reči mogli bi da eliminišemo uticaj uopštenih reči koje
nam, iako imaju visoku frekvenciju, ne daju korisne informacije o kritici.     
'''










