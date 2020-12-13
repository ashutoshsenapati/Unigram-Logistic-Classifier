import re
import sys
import nltk
import numpy as np
import csv

nltk.download('wordnet')
#np.set_printoptions(threshold=sys.maxsize)
#nltk.download('averaged_perceptron_tagger')
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from nltk import pos_tag
from nltk.corpus import wordnet
from itertools import chain
from sklearn.linear_model import LogisticRegression


negation_words = set(['not', 'no', 'never', 'nor', 'cannot'])
negation_enders = set(['but', 'however', 'nevertheless', 'nonetheless'])
sentence_enders = set(['.', '?', '!', ';'])


# Loads a training or test corpus
# corpus_path is a string
# Returns a list of (string, int) tuples
from typing import List
from typing import Tuple
def load_corpus(corpus_path: str) -> List[Tuple[List[str],int]]:
    final_list = []
    with open(corpus_path, encoding = 'latin') as file:
        data = file.read()
        sentences = data.split("\n")
  
        for sentence in sentences:
          lis = sentence.split("\t")
          if(len(lis) <2):
            continue
          else:
            strin = lis[0]
            num = int(lis[1])
            tup = (strin.split(" "),num)
            final_list.append(tup)
    return final_list


# Checks whether or not a word is a negation word
# word is a string
# Returns a boolean
def is_negation(word):
    if word in negation_words or word.endswith("n't"):
        return True
    else :
        return False

# Modifies a snippet to add negation tagging
# snippet is a list of strings
# Returns a list of strings
def tag_negation(snippet):
    position_list = nltk.pos_tag(snippet)
    index = -1
    compare_words = set(["JJR", "RBR"])

    index+=1
    while index <= len(position_list)-1:
        if not is_negation(position_list[index][0]):
            index+=1
        else:
            break

    if index <= len(position_list) - 2:
        if not position_list[index + 1][0] == "only":
            second = 0
            while second < len(position_list[index + 1:]):
                if (position_list[(index + 1) + second][0] in sentence_enders) or (position_list[(index + 1) + second][0] in negation_enders) or (position_list[(index + 1) + second][1] in compare_words):
                    break
                else:
                    snippet[(index + 1) + second] = "NOT_" + position_list[(index + 1) + second][0]
                    second = second + 1
    return snippet

# Assigns to each unigram an index in the feature vector
# corpus is a list of tuples (snippet, label)
# Returns a dictionary {word: index}
def get_feature_dictionary(corpus):
    index = 0
    dict = {}

    for tupl in corpus:
        for word in tupl[0]:
            if word not in dict:
                dict[word] = int(index)
                index+=1
    return dict
    

# Converts a snippet into a feature vector
# snippet is a list of tuples (word, pos_tag)
# feature_dict is a dictionary {word: index}
# Returns a Numpy array
def vectorize_snippet(snippet, feature_dict):
    arr = np.zeros(len(feature_dict),dtype = 'int32')
    for str in snippet:
        if str not in feature_dict:
            continue
        val = feature_dict[str]
        arr[val]+=1
    return arr

# Trains a classification model (in-place)
# corpus is a list of tuples (snippet, label)
# feature_dict is a dictionary {word: label}
# Returns a tuple (X, Y) where X and Y are Numpy arrays
def vectorize_corpus(corpus, feature_dict):
    n = len(corpus)
    d = len(feature_dict)
    X = np.empty([n,d], dtype = 'int32')
    Y = np.empty(n,dtype = 'int32')
    i=0
    for tupl in corpus:
        arr = vectorize_snippet(tupl[0], feature_dict)
        X[i] = arr
        Y[i] = tupl[1]
        i+=1
    return (X,Y)

# Performs min-max normalization (in-place)
# X is a Numpy array
# No return value
def normalize(X):
     for col in range(X.shape[1]):
        max_value = np.amax(X[:,col])
        min_value = np.amin(X[:,col])

        if max_value > 0 and min_value != max_value:
            X[:,col] = (X[:,col] - min_value)/(max_value - min_value)


# Trains a model on a training corpus
# corpus_path is a string
# Returns a LogisticRegression
def train(corpus_path):
    corpus = load_corpus(corpus_path)
    ngtd_corpus = []
    for tup in corpus:
      ngtd_corpus.append((tag_negation(tup[0]),tup[1]))
    feat_dict = get_feature_dictionary(ngtd_corpus)
    X = vectorize_corpus(corpus,feat_dict)[0]
    Y = vectorize_corpus(corpus,feat_dict)[1]
    model = LogisticRegression(random_state=0).fit(X,Y)
    print(model.score(X,Y))
    return (model, feat_dict)
    
# Calculate precision, recall, and F-measure
# Y_pred is a Numpy array
# Y_test is a Numpy array
# Returns a tuple of floats
def evaluate_predictions(Y_pred, Y_test):
    TP = 0
    FP = 0
    FN = 0

    for i in range(len(Y_test)): 
        if Y_test[i]==Y_pred[i]==1:
           TP += 1
        if Y_pred[i]==1 and Y_test[i]!=Y_pred[i]:
           FP += 1
        if Y_pred[i]==0 and Y_test[i]!=Y_pred[i]:
           FN += 1
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    F1 = 2*P*R / (P+R)
    return(P,R,F1)

# Evaluates a model on a test corpus and prints the results
# model is a LogisticRegression
# corpus_path is a string
# Returns a tuple of floats
def test(model, feature_dict, corpus_path):
    corpus = load_corpus(corpus_path)
    ngtd_corpus = []
    for tup in corpus:
      ngtd_corpus.append((tag_negation(tup[0]),tup[1]))
    d = len(feature_dict)
    n= len(ngtd_corpus)

    X = np.empty([n,d])
    
    Y = np.empty(n)
    for i,tup in enumerate(ngtd_corpus):
        for word in tup[0]:
            if word in feature_dict:
                X[i][feature_dict[word]] = 1
        Y[i] = tup[1]
    predY = model.predict(X)
    return evaluate_predictions(predY,Y)


# Selects the top k highest-weight features of a logistic regression model
# logreg_model is a trained LogisticRegression
# feature_dict is a dictionary {word: index}
# k is an int

def get_top_features(logreg_model, feature_dict, k=1):
    # pass
    ''' takes a trained LogisticRegression model and an int k and returns a list of length k containing tuples (word, weight) '''
    logistic_regression_coefficient = logreg_model.coef_
    updated_logreg = list()
    lr_updated = logistic_regression_coefficient[0]
    
    coef_idx = 0
    for coef in lr_updated:
        updated_logreg.append(tuple((coef_idx, coef)))
        coef_idx = coef_idx + 1
    
    sorted_logreg = sorted(updated_logreg, key=lambda x: abs(x[1]),reverse=True)
    sorted_idx = 0

    for weight_tuple in sorted_logreg:
        logreg_idx = weight_tuple[0]
        
        if logreg_idx < len(feature_dict):
            vocab = list(feature_dict)[logreg_idx]
            sorted_logreg[sorted_idx] = (vocab,weight_tuple[1])
            sorted_idx = sorted_idx + 1
        elif logreg_idx == len(feature_dict):
            sorted_logreg[sorted_idx] = ("activeness",weight_tuple[1])
        elif logreg_idx == len(feature_dict) + 1:
            sorted_logreg[sorted_idx] = ("evaluation", weight_tuple[1])
        elif logreg_idx == len(feature_dict) + 2:
            sorted_logreg[sorted_idx] = ("imagery",weight_tuple[1])
    return sorted_logreg[:k]

def main(args):
    model, feature_dict = train('train.txt')

    print(test(model, feature_dict, 'test.txt'))

    weights = get_top_features(model, feature_dict)
    for weight in weights:
        print(weight)
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
