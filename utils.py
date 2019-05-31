# lots of import for checking the environment
import numpy as np
import scipy
import sklearn
import modAL
import tqdm
import lime
import random

# actual imports
import matplotlib.pyplot as plt
import warnings
from sklearn.datasets import fetch_20newsgroups
from typing import Tuple, List, Any, Union, Callable
from flair.data import Token, Sentence
import pandas as pd



def load_news20(real: bool = False) -> Tuple[Tuple[list, list], Tuple[list, list], List[str]]:
    """
    Loads the 20 News Group dataset split by train and test as a raw text with class names,
    see: http://qwone.com/~jason/20Newsgroups/ for details.
    :param real: bool, default False
        Whether to use the `real` dataset, with headers, footers and quotes stripped

    :return: tuple (train set, test set, class names)
        Train and test set as tuples (data, target) and the class names is a list
    """

    if real:
        train_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
        test_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    else:
        train_data = fetch_20newsgroups(subset='train')
        test_data = fetch_20newsgroups(subset='test')

    # extract class names
    class_names = train_data['target_names']

    # extract raw data and labels
    X_raw_train, y_train_full = train_data['data'], train_data['target']
    X_raw_test, y_test_full = test_data['data'], test_data['target']

    # reformat class names to more readable
    class_names = [x.split('.')[-1] if 'misc' not in x else '.'.join(x.split('.')[-2:]) for x in class_names]
    class_names[3] = 'pc.hardware'
    class_names[4] = 'mac.hardware'
    class_names[5] = 'ms-windows.x'

    return (X_raw_train, y_train_full), (X_raw_test, y_test_full), class_names


def replace_token(sentence: Sentence, token: Token, new_word: str = ""):
    """Replaces token with word."""
    if new_word == "":
        sentence.tokens.remove(token)
    else:
        token.text = new_word
        
        
def most_important_words_black(text, label:int, pipeline: Callable[[List[str]], np.array]) -> List[Tuple[str, float]]:
    """Returns list of words sorted descending by importance in text and loss in true class probability."""
    sentence = Sentence(text, True)
    tokens = sentence.tokens
    if len(tokens) == 1:
        return [(text,1)]
    true_probability = [None]*len(tokens)
    new_texts = [""]*len(tokens)
    
    # create list of modified sentences with removed words 
    for i, token in enumerate(tokens):
        sentence_tmp = Sentence()
        for t in tokens:
            sentence_tmp.add_token(t)
        replace_token(sentence_tmp, token)
        new_texts[i] = sentence_tmp.to_plain_string()
    
    output = pipeline(new_texts)
    org_prob = pipeline([text])[0][label]
    
    # probability of of true label with removed word
    for i, o in enumerate(output):
        true_probability[i] = o[label]
        
    # return sorted ascending by probability is equivalent of sorted descending by importance
    return [(x.idx, x.text, org_prob - p)for p ,x in sorted(zip(true_probability,tokens), key=lambda pair: pair[0])]


def change_most_important_word(text, label:int, pipeline: Callable[[List[str]], np.array], corruptor) -> str:
    """Returns sentece with corrupt most important word."""
    sentence = Sentence(text, True)
    tokens = sentence.tokens
    if len(tokens) == 1:
        return [(text,1)]
    true_probability = [None]*len(tokens)
    new_texts = [""]*len(tokens)
    
    # create list of modified sentences with removed words 
    for i, token in enumerate(tokens):
        sentence_tmp = Sentence()
        for t in tokens:
            sentence_tmp.add_token(t)
        replace_token(sentence_tmp, token)
        new_texts[i] = sentence_tmp.to_plain_string()
    
    output = pipeline(new_texts)
    org_prob = pipeline([text])[0][label]
    
    # probability of of true label with removed word
    for i, o in enumerate(output):
        true_probability[i] = o[label]

    most_imp_index = np.argmin(true_probability)
    tokens[most_imp_index].text = corruptor(tokens[most_imp_index].text)
	
    return sentence.to_plain_string()


def change_random_word(text,  corruptor) -> str:
    """Returns sentece with corrupt random word."""
    sentence = Sentence(text, True)
    tokens = sentence.tokens
	
    random_token = random.choice(tokens)
    random_token.text = corruptor(random_token.text)
	
    return sentence.to_plain_string()
     	
	
