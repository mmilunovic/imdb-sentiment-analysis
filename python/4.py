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
    
# Contains all operations related to feature extraction and
# data split
class Dataset:
    def __init__(self, path, nb_words, test_ratio, y):
        
        # Loads both neg and pos reviews to corpus
        neg_corpus = read_folder(path+"\\neg")
        pos_corpus = read_folder(path+"\\pos")
        for pos in pos_corpus:
            neg_corpus.append(pos)
        corpus = neg_corpus
        
        # Prepare data for further analysis
        clean_corpus = self.prepare_corpus(corpus)
        
        # Create BoW feature vector from clean_corpus
        x = self.create_bow(clean_corpus)
        
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
        print('Cleaning the corpus...')
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
        print("Computing BoW histogram...")
        freq = FreqDist([w for rew in clean_corpus for w in rew])
        best_words, _ = zip(*freq.most_common(nb_words)) # Most used nb_words(1000) words
        x = []
        for rew in clean_corpus:
            bow = dict()
            for i in range(nb_words):
                cnt = rew.count(best_words[i])
                if cnt > 0:
                    bow[i] = cnt
            x.append(bow)
        return x
    
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


path_to_folder = "D:\\RAF\\6. Semestar\\Mašinsko učenje\\raf-ml\\Domaci 1\\ml_d1_x_y_z\\data\\imdb"
samples = 1250 # Use all data
nb_words = 1000 # As said in statement
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

# Train the model with train data
print("Tranning...")
model = MultinomialNaiveBayes(nb_classes, nb_words, pseudocount)
model.fit(dataset.train)

preds_train = model.predict(dataset.train['x'])
acc_train = calc_acc(dataset.train['y'], preds_train)
print("Train set accuracy: {:.2f}".format(round(acc_train*100)))

# Test the model on remaingin 20% of reviews
predictions = model.predict(dataset.test['x'])
acc_test = calc_acc(dataset.test['y'], predictions)
print("Test set accuracy: {:.2f}%".format(round(acc_test*100)))