#-*- coding: utf-8 -*-
from transformers import AutoTokenizer
from sentence_transformers import models, SentenceTransformer, util
import scipy
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle as pkl
from konlpy.tag import Okt
import re


#피클로 기존에 저장되있는 판시사항들 리스트 불러오기
with open("master_items.pickle", "rb") as f:
    master_items = pkl.load(f)


# 모델 불러오기 및 설정
word_embedding_model = models.Transformer("monologg/koelectra-base-v3-discriminator")
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True, pooling_mode_cls_token=False, pooling_mode_max_tokens=False)
model=SentenceTransformer(modules=[word_embedding_model, pooling_model])

# # 판시사항 문장 표현 얻기
# dataFileName='./df_summary.csv'
# df = pd.read_csv(dataFileName)
# df = df.where((pd.notnull(df)), '')

# master_items=df['items'].values.tolist() # 데이터프레임의 판시사항 열을 리스트로 변환
# with open("master_items.pickle", "wb") as f:
#     pkl.dump(master_items, f)

# # 전처리
# master_items_processed = []
# okt = Okt()
# for item in master_items:
#     nonhangul = re.compile('[^ ㄱ-ㅣ가-힣]+') # 한글과 띄어쓰기를 제외한 모든 글자
#     item = nonhangul.sub(' ', item) # 한글과 띄어쓰기를 제외한 모든 부분을 제거
# #     # # 형태소 분석
# #     # malist = okt.pos(item, norm=True, stem=True)
# #     # # 필요한 어구만 대상으로 하기
# #     # sentences = []
# #     # r = []
# #     # for word in malist:
# #     #     # 어미/조사/구두점 등은 대상에서 제외 
# #     #     if not word[1] in ["Josa", "Eomi", "Punctuation"]:
# #     #         r.append(word[0])
# #     # sentences.append(" ".join(r))
#     master_items_processed.append(master_items)

# #master_items_encoded = model.encode(master_items, convert_to_tensor=True) # 판시사항들의 문장 표현 구하기
# master_items_encoded = model.encode(master_items)
# with open("electra_master_items_encoded_no_re.pickle", "wb") as f:
#     pkl.dump(master_items_encoded, f)

with open("electra_master_items_encoded_no_re.pickle", "rb") as f:
    master_items_encoded = pkl.load(f)

# 임의의 Doc 문장 표현 얻기
#input_doc = input("뉴스 원문을 입력하세요:")
input_doc = "말기 암 투병 중인 50대가 생활고를 비관해 지적장애가 있는 딸을 살해하고 극단적인 선택을 시도했다가 경찰에 자수했다."
# nonhangul = re.compile('[^ ㄱ-ㅣ가-힣]+') # 한글과 띄어쓰기를 제외한 모든 글자
# okt = Okt()
# input_doc_processed = []
# item2 = nonhangul.sub(' ', input_doc) # 한글과 띄어쓰기를 제외한 모든 부분을 제거
# # # 형태소 분석
# # malist2 = okt.pos(item2, norm=True, stem=True)
# # # 필요한 어구만 대상으로 하기
# # r2 = []
# # for word in malist2:
# #     # 어미/조사/구두점 등은 대상에서 제외 
# #     if not word[1] in ["Josa", "Eomi", "Punctuation"]:
# #         r2.append(word[0])
# # input_doc_processed.append(" ".join(r2))

# #input_doc_encoded = model.encode(input_doc, convert_to_tensor=True)
input_doc_encoded = model.encode(input_doc)
print(input_doc_encoded)

#유사도 계산 (코사인 유사도 구하기)
similarity = util.pytorch_cos_sim(input_doc_encoded,master_items_encoded)
print(similarity)
#sklearn으로 코사인 유사도 구하기
# cos_sim = []
# for i in master_items_encoded:
#     similarity=cosine_similarity(input_doc_encoded,i)
#     cos_sim.append(similarity)
# max_num = max(cos_sim)
# max_num_index=cos_sim.index(max_num)

#판시사항 출력
print(master_items[np.argmax(similarity)])
#print(master_items[max_num_index])
print()

# #가장 유사한 5개의 판시사항 추출
# ordered_similarity = np.argsort(-similarity) # 내림차순 정렬
# for i in ordered_similarity[:5]:
#     print(master_items[i])

# model2=SentenceTransformer('xlm-r-base-en-ko-nli-ststb')
# master_items_encoded2 = model2.encode(master_items)
# input_doc_encoded2 = model2.encode(input_doc)
# similarity2 = util.pytorch_cos_sim(input_doc_encoded2,master_items_encoded2)
# print(master_items[np.argmax(similarity2)])