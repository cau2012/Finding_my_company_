import sys
sys.path.append('C:\\Users\\dlfdus\\PycharmProjects\\mypacakge\\')
from corpus import DoublespaceLineCorpus
from corpus import DoublespaceLineCorpusWithList
from tqdm import tqdm_notebook
import requests
import pandas as pd
import re
import pickle
from collections import Counter

import pycrfsuite
from pycrfsuite_spacing import TemplateGenerator
from pycrfsuite_spacing import CharacterFeatureTransformer
from pycrfsuite_spacing import PyCRFSuiteSpacing
from soynlp.tokenizer import LTokenizer
import numpy as np

with open('data/tmp/tokenized_reviews.pkl', 'rb') as f:
    tokenized_reviews = pickle.load(f)
with open('data/tmp/word_dictionary.pkl', 'rb') as f:
    word_dictionary = pickle.load(f)
with open('data/tmp/jobplanet_cohesionscore.pkl', 'rb') as f:
    cohesion_scores = pickle.load(f)
ltokenizer = LTokenizer(scores=cohesion_scores)

to_feature = CharacterFeatureTransformer(TemplateGenerator(begin=-2,
                                                           end=2,
                                                           min_range_length=3,
                                                           max_range_length=3))
model_path = 'data/tmp/package_test.crfsuite'
correct = PyCRFSuiteSpacing(to_feature)
correct.load_tagger(model_path)
with open('data/tmp/lr.pkl', 'rb') as f:
    model = pickle.load(f)
with open('data/tmp/nouns.pkl', 'rb') as f:
    nouns = pickle.load(f)


def get_noun(doc, nouns=nouns):
    a = ltokenizer.tokenize(doc)
    noun_list = []
    for indx, i in enumerate(a):
        if i in nouns:
            noun_list.append(i)
    return noun_list

def tokenizer(comment, only_noun = False):
    comment = correct(comment)
    tokenized = ''
    for word in ltokenizer.tokenize(comment):
        if word in word_dictionary and len(word)>1:
            tokenized += word
            tokenized += ' '
    tokenized = tokenized.strip()
    if only_noun == True:
        tokenized = get_noun(tokenized)
    return tokenized


def sentiment_analysis(model, vectorizer):
    comment = input()
    tokenized_comment = [tokenizer(comment)]
    #print(' -> ',tokenizer(comment))
    vectorized_coemment = vectorizer.transform(tokenized_comment).toarray()
    predict = model.predict(vectorized_coemment)
    print(' ')
    if predict[0] == 0:
        #print('긍정 문장')
        return '긍정 문장'
    else:
        #print('부정 문장')
        return '부정 문장'