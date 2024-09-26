from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from konlpy.tag import Okt

import pandas as pd
import numpy as np
import pickle

# PREPROC #########################################################################
print("preproc loading...")
with open('preproc_sen2vec.preproc', 'rb') as f:
    preproc = pickle.load(f) # (사건번호, 단어 리스트) 튜플의 리스트
df_preproc = pd.DataFrame(preproc, columns = ['case_full_no', 'gists'])
for i in range(len(df_preproc)):
    df_preproc.iloc[i]['gists'] = ' '.join(df_preproc.iloc[i]['gists']) 
print(df_preproc.head(5))

# DOC2VEC #########################################################################
print("\n# doc2vec retrieving")
with open('embed_doc2vec.model', 'rb') as f:
    doc2vec_model = pickle.load(f)

print("# doc2vec example")
# print(doc2vec_model.wv.most_similar(positive = ["채무불이행", "이행불능"], negative = ["변제"]))
# print(doc2vec_model.wv.most_similar(positive = ["변호사", "사기"], negative = ["무죄"]))
v_list =["아파트", "분양"]
print(v_list)
rlist = []
df = df_preproc[df_preproc['gists'].str.contains(' '.join(v_list))] # 판결요지에서 어구 검색
if len(df) != 0:
    if len(df) > 5:
        itr = 5
    else:
        itr = len(df)
    for i in range(itr):
            
        # 기준판례
        print('\n')
        print("---기준판례 사건번호/판결요지")
        print(df.iloc[i][0])
        print()
        print(df.iloc[i][1])

        # 유사판례 #########################################################################
        print("\n---유사판례 사건번호")
        result = doc2vec_model.dv.most_similar(positive = [doc2vec_model.infer_vector(df.iloc[i][1].split(" "))], topn=5)
        print('\n')
        print(result)
        rlist.append(result)

print("\n# word2vec 256 retrieving")
with open('embed_word2vec_256.model', 'rb') as f:
    model = pickle.load(f)

# WORD2VEC 256 #########################################################################
print("# word2vec 256 example")
print(model.wv.most_similar(positive = ["채무불이행", "이행불능"], negative = ["변제"]))
print(model.wv.most_similar(positive = ["변호사", "사기"], negative = ["무죄"]))

# WORD2VEC #########################################################################
print("\n# word2vec retrieving")
with open('embed_word2vec.model', 'rb') as f:
    model = pickle.load(f)

print("# word2vec example")
print(model.wv.most_similar(positive = ["채무불이행", "이행불능"], negative = ["변제"]))
print(model.wv.most_similar(positive = ["변호사", "사기"], negative = ["무죄"]))
print(model.wv["지체"])
print(type(model.wv["지체"]))

print(list(model.wv.key_to_index.keys())[:50])
# one_hot_matrix = np.zeros((nda.shape[0], 3), dtype=np.float32)
# print(one_hot_matrix)
# one_hot_matrix[:,0] = nda
# print(one_hot_matrix)
# print(model.vector_size)
# DOC2VEC
# print("# doc2vec example")
# result = doc2vec_model.dv.most_similar(positive = [doc2vec_model.infer_vector('이행불능 이행지체'.split(" "))], topn=10)
# print(result)

