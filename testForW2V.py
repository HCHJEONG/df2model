# 단어 임베딩 for 검색어 추천
from collections import Counter
from gensim.models import word2vec
from gensim import utils
from konlpy.tag import Okt, Kkma
from multiprocessing import cpu_count

import json, requests
import re
import os
import numpy as np
import glob
import pickle
import tqdm
import parmap
import pandas as pd


# PREPROC #########################################################################
# print("preproc loading...")
# with open('model\\word2vec_from_listForCaseSentenceForreasoning_with_Nori.preproc', 'rb') as f:
#     preproc = pickle.load(f) # (사건번호, 단어 리스트) 튜플의 리스트

# print(preproc[:5])

# WORD2VEC #########################################################################
print("\n# word2vec 이유 retrieving")
with open('model\\word2vec_from_listForCaseSentenceForreasoning_with_512.model', 'rb') as f:
    model1 = pickle.load(f)
print("\n# word2vec 요지 retrieving")
with open('model\\word2vec_from_listForCaseSentenceForgists_with_512.model', 'rb') as f:
    model2 = pickle.load(f)

print("\n# word2vec 이유 example")
print(model1.wv.most_similar(positive = ["부정경쟁", "영업비밀"]))
print(model1.wv.most_similar(positive = ["재건축", "안전진단"]))
print(model1.wv.most_similar(positive = ["채무불이행", "이행불능"], negative = ["변제"]))
print(model1.wv.most_similar(positive = ["변호사", "사기"], negative = ["무죄"]))
print("\n# word2vec 요지 example")
print(model2.wv.most_similar(positive = ["부정경쟁", "영업비밀"]))
print(model2.wv.most_similar(positive = ["재건축", "안전진단"]))
print(model2.wv.most_similar(positive = ["채무불이행", "이행불능"], negative = ["변제"]))
print(model2.wv.most_similar(positive = ["변호사", "사기"], negative = ["무죄"]))
# print(model.wv["지체"])
# print(type(model.wv["지체"]))

# print(list(model.wv.key_to_index.keys())[:50])