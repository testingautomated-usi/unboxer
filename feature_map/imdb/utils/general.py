import matplotlib
matplotlib.use('Agg')
import pandas as pd
import nltk
# For Python 3.6 we use the base keras
from tensorflow import keras
import numpy as np
import random
from difflib import SequenceMatcher
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.tokenize.treebank import TreebankWordDetokenizer

import Levenshtein as lev

INDEX_FROM=3   # word index offset


def untokenize(vector):
    return TreebankWordDetokenizer().detokenize(vector)    

def find_adjs(text):
    tokenized_text = word_tokenize(text)
    word_tags = nltk.pos_tag(tokenized_text)
    adjs_advs = [i for i in range(0,len(word_tags)) if word_tags[i][1] in ['JJ', 'JJR', 'JJS']]
    return word_tags, adjs_advs


def get_synonym(word):
    word = word.lower()
    synonyms = []
    synsets = wordnet.synsets(word)
    if (len(synsets) == 0):
        return []
    for synset in synsets:
        lemma_names = synset.lemma_names()
        for lemma_name in lemma_names:
            lemma_name = lemma_name.lower().replace('_', ' ')
            if (lemma_name != word and lemma_name not in synonyms):
                synonyms.append(lemma_name)
    if len(synonyms) > 0:
        sword = random.choice(synonyms)
        return sword
    else:
        return None


def listToString(s): 
    
    # initialize an empty string
    str1 = s[0] 
    
    # traverse in the string  
    for ele in s[1:]: 
        if isinstance(ele, str):
            str1 += "." + ele  
    
    # return string  
    return str1 

def decode_imdb_reviews(embd):

    word_to_id = keras.datasets.imdb.get_word_index()
    word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    word_to_id["<UNUSED>"] = 3

    id_to_word = {value:key for key,value in word_to_id.items()}
    text = ' '.join(id_to_word[id] for id in embd)

    return text



def compute_sparseness(map, x):
    n = len(map)
    # Sparseness is evaluated only if the archive is not empty
    # Otherwise the sparseness is 1
    if (n == 0) or (n == 1):
        sparseness = 0
    else:
        sparseness = density(map, x)
    return sparseness

def get_neighbors(b):
    neighbors = []
    neighbors.append((b[0], b[1]+1))
    neighbors.append((b[0]+1, b[1]+1))
    neighbors.append((b[0]-1, b[1]+1))
    neighbors.append((b[0]+1, b[1]))
    neighbors.append((b[0]+1, b[1]-1))
    neighbors.append((b[0]-1, b[1]))
    neighbors.append((b[0]-1, b[1]-1))
    neighbors.append((b[0], b[1]-1))

    return neighbors

def density(map, x):
    b = x.features
    density = 0
    count = 0
    neighbors = get_neighbors(b)
    for neighbor in neighbors:
        if neighbor not in map:
            density += 1
    return density



def get_distance(v1, v2):
    return lev.distance(v1,v2)



def rescale_map(features, perfs, new_min_1, new_max_1, new_min_2, new_max_2):
    if new_max_1 > 25:
        shape_1 = 25
    else:
        shape_1 = new_max_1 + 1
    
    if new_max_2 > 25:
        shape_2 = 25
    else:
        shape_2 = new_max_2 + 1

    output = dict()

    original_bins1 = np.linspace(new_min_1, new_max_1, shape_1)
    original_bins2 = np.linspace(new_min_2, new_max_2, shape_2)

    for key, value in perfs.items():
        i = key[0]
        j = key[1]
        if i < new_max_1 and j < new_max_2:
            new_i = np.digitize(i, original_bins1, right=False)
            new_j = np.digitize(j, original_bins2, right=False)
            if (new_i, new_j) not in output or value < output[(new_i, new_j)]:
                output[(new_i, new_j)] = value
    return output

# Useful function that shapes the input in the format accepted by the ML model.




def missing(s: pd.Series):
    """
    :return: The number of NaN values in a series
    """
    return s.isnull().sum()

from feature_map.imdb.predictor import Predictor


def text_to_vec(text):
    """
    converst text to vec using bag of words
    :param text: The text to convert
    :return: The generated vector
    """
    text_matrix = Predictor.tokenizer.texts_to_matrix(text, mode="binary")
    return text_matrix