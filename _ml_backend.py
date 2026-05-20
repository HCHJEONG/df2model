
import os
import random
import pickle
import pandas as pd
from collections import Counter
from pprint import pprint
from dotenv import load_dotenv
load_dotenv()

from elasticsearch import Elasticsearch

import requests
import json
import re

from flask import Flask, jsonify, request
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS

import gensim 
# (old) 3.8.1 version with python 3.8.11 / 3.7.16
# (202412~) 4.3.0 version with python 3.8.11 # https://github.com/piskvorky/gensim/wiki/Migrating-from-Gensim-3.x-to-4
# from gensim.models import word2vec

import nltk # with python 3.8.11 / 3.7.16
try:
  nltk.data.find('tokenizers/punkt')
except LookupError:
  nltk.download('punkt', quiet=True)

import networkx as nx
from networkx.algorithms import traversal

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM # t5 False 현재 미사용

numOfrecommended = 15 # api 1-4/2/3/4/5/6


ELASTIC_CA_CERTS_FILEPATH = os.getenv("ELASTIC_CA_CERTS_FILEPATH")
ELASTIC_ID = os.getenv("ELASTIC_ID")
ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")
ELASTIC_HOST = os.getenv("ELASTIC_HOST") or ""
ELASTIC_PORT = os.getenv("ELASTIC_PORT") or ""

es = None
if ELASTIC_HOST and ELASTIC_PORT:
  if "https" in ELASTIC_HOST:
    es = Elasticsearch(
      ELASTIC_HOST + ":" + ELASTIC_PORT,
      ca_certs=ELASTIC_CA_CERTS_FILEPATH,
      http_auth=(ELASTIC_ID, ELASTIC_PASSWORD)
    )
  else:
    es = Elasticsearch(
      ELASTIC_HOST + ":" + ELASTIC_PORT,
    ) 
print("\nelasticsearch DB information: ")   
try: 
  if es is None:
    raise RuntimeError("ELASTIC_HOST/ELASTIC_PORT is not configured")
  print(es.info())
  print(es.cat.indices())
except Exception as e:
  print('No ES client process running in this local machine...\n')
  print(e)
print()


model_wv = None
W2VMODEL_FILEPATH = './model/word2vec_from_listForCaseSentenceForreasoning_with_Nori_with_512.model'
model_wv_vsize = 512
word2vec_wordList = []

model_dv = None
D2VMODEL_FILEPATH = './model/doc2vec_from_CorpusIGR_with_Nori_with_512.model'
df_d2v_tagtable = None
D2V_TAGTABLE = './model/doc2vec_from_CorpusIGR_with_Nori_with_512.preproc'

model_dv_ccase = None
D2VMODEL_CCASE_FILEPATH = './model/doc2vec_from_CCaseIGBA_with_Nori_with_512.model'
df_d2v_ccase_tagtable = None
D2V_CCASE_TAGTABLE = './model/doc2vec_from_CCaseIGBA_with_Nori_with_512.preproc'

model_dv_csummary = None
D2VMODEL_CSUMMARY_FILEPATH = './model/doc2vec_from_SummaryIGA_with_Nori_with_512.model'
df_d2v_csummary_tagtable = None
D2V_CSUMMARY_TAGTABLE = './model/doc2vec_from_SummaryIGA_with_Nori_with_512.preproc'

GRAPHSUMMARY_FILEPATH = './model/graphSummary.pickle'
graphSummary = None
graphSummaryReversed = None
GRAPHCORPUS_FILEPATH = './model/graphCorpus.pickle'
graphCorpus = None

# 모델 성능에 문제가 있어 미사용 start
phrase = False
model_dvp = None
phraseserial = None
vsize = 512
cname = 'cname'
engine = 'Nori'
source = 'SummaryGsPhrases'
modelpfilepath = './model/doc2vec_from_'+source+'_with_'+engine+'_with_'+str(vsize)+'.model'
phraseserialfilepath = './model/doc2vec_from_'+source+'_with_'+engine+'_with_'+str(vsize)+'.phraseserial'

t5 = False
model_t5_dir = "lcw99/t5-base-korean-text-summary"
max_input_length = 512
model_t5_pretrained = "./model/t5-base-korean-text-summary"
# 모델 성능에 문제가 있어 미사용 end


# model / preproc loading
print("models loading...\n")    
print("gensim version: ")
print(gensim.__version__)
print()

print("w2v model loading...")
# with open('../df2model/model/embed_word2vec.model', 'rb') as f:
# C:\Users\hcjeo\VSCodeProjects\df2model\model\embed_word2vec_from_reasoning_512.model    
try:
  with open(W2VMODEL_FILEPATH, 'rb') as f:
      model_wv = pickle.load(f)
  print("word2vec model loaded: ")
  print(dir(model_wv))
except:
  print("There is no saved model locally...")
print()

print("word2vec model word list retrieving...")
try:
  # word2vec model word list retrieving
  # print(dir(model_wv.wv))
  # 'add_lifecycle_event', 'add_vector', 'add_vectors', 'allocate_vecattrs', 'closer_than', 'cosine_similarities', 'distance', 'distances', 'doesnt_match', 'evaluate_word_analogies', 'evaluate_word_pairs', 'fill_norms', 'get_index', 'get_normed_vectors', 'get_vecattr', 'get_vector', 'has_index_for', 'index2entity', 'index2word', 'init_sims', 'intersect_word2vec_format', 'load', 'load_word2vec_format', 'log_accuracy', 'log_evaluate_word_pairs', 'most_similar', 'most_similar_cosmul', 'most_similar_to_given', 'n_similarity', 'rank', 'rank_by_centrality', 'relative_cosine_similarity', 'resize_vectors', 'save', 'save_word2vec_format', 'set_vecattr', 'similar_by_key', 'similar_by_vector', 'similar_by_word', 'similarity', 'similarity_unseen_docs', 'sort_by_descending_frequency', 'unit_normalize_all', 'vector_size', 'vectors', 'vectors_for_all', 'vectors_norm', 'vocab', 'wmdistance', 'word_vec', 'words_closer_than'
  # Use KeyedVector's .key_to_index dict, .index_to_key list, and methods .get_vecattr(key, attr) and .set_vecattr(key, attr, new_val) 
  # print(type(model_wv.wv.get_vecattr()))
  # print(model_wv.wv.index_to_key[:20])
  # word2vec_wordList = model_wv.wv.index_to_key
  # word2vec_wordList = list(model_wv.wv.vocab.keys())
  word2vec_wordList = list(model_wv.wv.key_to_index.keys())
  print("word2vec model word list retrieved: ")
  print(word2vec_wordList[:30])
except Exception as e:
  print("There is no wordlist in the model...")
  print("cause: ")
  print(e)    
  # The vocab attribute was removed from KeyedVector in Gensim 4.0.0.
  # Use KeyedVector's .key_to_index dict, .index_to_key list, and methods .get_vecattr(key, attr) and .set_vecattr(key, attr, new_val) instead.
  # See https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4
print()

print("doc2vec model loading...")
try:
    with open(D2VMODEL_FILEPATH, 'rb') as f:
        model_dv = pickle.load(f)
    print("doc2vec model loaded")
except:
    print("There is no saved doc2vec model file locally...")
print()

print("doc2vec preproc loading...")
# with open('../df2model/model/preproc_doc2vec.preproc', 'rb') as f:
#     preproc = pickle.load(f) # (사건번호, 단어 리스트) 튜플의 리스트
# df_d2v_tagtable = pd.DataFrame(preproc, columns = ['case_full_no', 'reasoning'])
# for i in tqdm(range(len(df_d2v_tagtable))):
#     df_d2v_tagtable.iloc[i]['reasoning'] = ' '.join(df_d2v_tagtable.iloc[i]['reasoning']) 
# with open('../df2model/model/preproc_doc2vec_joined_reason.pickle', 'wb') as f:
#     pickle.dump(df_d2v_tagtable, f)
try:
  # should be columns = ['case_full_no', 'reasoning']
  with open(D2V_TAGTABLE, 'rb') as f:
  # with open('./model/preproc_doc2vec_joined_reason.pickle', 'rb') as f:
    d2v_tagtable = pickle.load(f) 
    df_d2v_tagtable = pd.DataFrame(d2v_tagtable, columns=['case_full_no','token_list'])
  print("preproc loaded: ")
  print(df_d2v_tagtable.head(5))
except:
    print("There is no saved preproc doc2vec joined reason pickle file locally...")
print()

print("doc2vec model ccase loading...")
try:
    with open(D2VMODEL_CCASE_FILEPATH, 'rb') as f:
        model_dv_ccase = pickle.load(f)
    print("doc2vec model ccase loaded")
except:
    print("There is no saved doc2vec ccase model file locally...")
print()

print("preproc ccase loading...")
# with open('../df2model/model/preproc_doc2vec.preproc', 'rb') as f:
#     preproc = pickle.load(f) # (사건번호, 단어 리스트) 튜플의 리스트
# df_d2v_tagtable = pd.DataFrame(preproc, columns = ['case_full_no', 'reasoning'])
# for i in tqdm(range(len(df_d2v_tagtable))):
#     df_d2v_tagtable.iloc[i]['reasoning'] = ' '.join(df_d2v_tagtable.iloc[i]['reasoning']) 
# with open('../df2model/model/preproc_doc2vec_joined_reason.pickle', 'wb') as f:
#     pickle.dump(df_d2v_tagtable, f)
try:
  # should be columns = ['case_full_no', 'reasoning']
  with open(D2V_CCASE_TAGTABLE, 'rb') as f:
  # with open('./model/preproc_doc2vec_joined_reason.pickle', 'rb') as f:
    d2v_ccase_tagtable = pickle.load(f) 
    df_d2v_ccase_tagtable = pd.DataFrame(d2v_ccase_tagtable, columns=['caseno','token_list'])
  print("preproc ccase loaded: ")
  print(df_d2v_ccase_tagtable.head(5))
except:
    print("There is no saved preproc doc2vec ccase joined reason pickle file locally...")
print()

print("doc2vec model csummary loading...")
try:
  with open(D2VMODEL_CSUMMARY_FILEPATH, 'rb') as f:
      model_dv_csummary = pickle.load(f)
  print("doc2vec model csummary loaded")
except:
  print("There is no saved doc2vec csummary model file locally...")
print()

print("preproc csummary loading...")
# with open('../df2model/model/preproc_doc2vec.preproc', 'rb') as f:
#     preproc = pickle.load(f) # (사건번호, 단어 리스트) 튜플의 리스트
# df_d2v_tagtable = pd.DataFrame(preproc, columns = ['case_full_no', 'reasoning'])
# for i in tqdm(range(len(df_d2v_tagtable))):
#     df_d2v_tagtable.iloc[i]['reasoning'] = ' '.join(df_d2v_tagtable.iloc[i]['reasoning']) 
# with open('../df2model/model/preproc_doc2vec_joined_reason.pickle', 'wb') as f:
#     pickle.dump(df_d2v_tagtable, f)
try:
  # should be columns = ['case_full_no', 'reasoning']
  with open(D2V_CSUMMARY_TAGTABLE, 'rb') as f:
  # with open('./model/preproc_doc2vec_joined_reason.pickle', 'rb') as f:
    d2v_csummary_tagtable = pickle.load(f) 
    df_d2v_csummary_tagtable = pd.DataFrame(d2v_csummary_tagtable, columns=['case_full_no_no','token_list'])
  print("preproc csummary loaded:")
  print(df_d2v_csummary_tagtable.head(5))
except:
    print("There is no saved preproc doc2vec csummary joined reason pickle file locally...")
print()


print('graph nexworkX loading...')
try:
  with open(GRAPHSUMMARY_FILEPATH, 'rb') as f:
    graphSummary = pickle.load(f)
    graphSummaryReversed = graphSummary.reverse()
  with open(GRAPHCORPUS_FILEPATH, 'rb') as f:
    graphCorpus = pickle.load(f)
  print("graph nexworkX loaded")
except:
  print("There is no saved nexworkX pickle locally...")
print()

if phrase:
  print('doc2vecmodel phrase loading...')
  try:
    with open(modelpfilepath, 'rb') as f:
      model_dvp = pickle.load(f)
    print("doc2vecmodel phrase loaded")
  except:
    print("There is no saved doc2vec model phrase locally...")
  print()

  print("preproc phrase loading...")
  try:
    # should be columns = ['case_full_no', 'reasoning']
    with open(phraseserialfilepath, 'rb') as f:
    # with open('./model/preproc_doc2vec_joined_reason.pickle', 'rb') as f:
      phraseserial = pickle.load(f) 
    print("preproc phrase loaded:")
    print(phraseserial[:5])
  except:
    print("There is no saved preproc doc2vec phrase pickle locally...")
  print()

if t5:
  print("model t5 and tokenizer loading...")
  try:
    tokenizer = AutoTokenizer.from_pretrained(model_t5_dir)
    tokenizer.save_pretrained(model_t5_pretrained)
    tokenizer = AutoTokenizer.from_pretrained(model_t5_pretrained)
    model_t5 = AutoModelForSeq2SeqLM.from_pretrained(model_t5_dir)
    model_t5.save_pretrained(model_t5_pretrained, from_pt=True)
    model_t5 = AutoModelForSeq2SeqLM.from_pretrained(model_t5_pretrained)
  except:
    print("There is no saved t5 model pretrained...")
print()


class HEALTHCHECK(Resource):    
    # GET      
    def get(self, ): 
      
      response = jsonify({"description":"penvot ml backend healthcheck", "result":"healthy"})
      response.status_code = 200 # or 400 or whatever
      return response
    
class ECASELIST():
  def __init__(self, id, pw):
    # 전자소송 로그인 id 및 pw 설정
    self.id = id
    self.pw = pw

  def get_client_cookies(self):
    # 비로그인 상태에서 로그인시 사용할 cookies 및 token 획득
    self.ecfs_home_url = "https://ecfs.scourt.go.kr/ecf/index.jsp#_"
    self.ecfs_home_headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br, zstd',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7,ja;q=0.6',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Host': 'ecfs.scourt.go.kr',
        'Pragma': 'no-cache',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'sec-ch-ua': '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"'
    }
    ecfs_home_res = requests.session().get(url=self.ecfs_home_url,headers=self.ecfs_home_headers)
    self.ecfs_JESSION_cookie = ecfs_home_res.cookies.get_dict()
    # token 획득하기
    token_regex = re.compile('du.c.TOKEN.*";')
    token = token_regex.findall(ecfs_home_res.text)
    token = re.findall("\".*\"",token[0])
    self.token = token[0].replace('"',"")
    
    self.ecfs_usermetatoken = self.token
    ecfs_WMONID_payload = {
    "_userMetaToken": self.ecfs_usermetatoken
    }
    ecfs_WMONID_headers = {
        'Accept': 'application/json, text/javascript, */*',
        'Accept-Encoding': 'gzip, deflate, br, zstd',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7,ja;q=0.6',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Host': 'ecfs.scourt.go.kr',
        'Origin': 'https://ecfs.scourt.go.kr',
        'Pragma': 'no-cache',
        'Referer': 'https://ecfs.scourt.go.kr/ecf/index.jsp',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'sec-ch-ua': '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"'
    }
    ecfs_WMONID_url = "https://ecfs.scourt.go.kr/ecf/ecf/ECF000/ECF010s01Cmd.ajax"
    client_login_res = requests.post(url=ecfs_WMONID_url,headers=ecfs_WMONID_headers, cookies=self.ecfs_JESSION_cookie, data=ecfs_WMONID_payload)
    self.ecfs_WMONID_cookie = client_login_res.cookies.get_dict()
    self.ecfs_JESSION_cookie.update(self.ecfs_WMONID_cookie)
    self.ecfs_cookies = self.ecfs_JESSION_cookie

    return self.ecfs_cookies
  
  def post_login_session(self):
    # cookies, token 과 함께 로그인정보(id, pw)를 전자소송서버에 POST
    self.ecfs_login_register_url = "https://ecfs.scourt.go.kr/ecf/ecf/ECF110/ECF110s01Cmd.ajax"
    self.ecfs_login_register_payload = {
        "lgnType": "G",
        "userId": self.id,
        "pw": self.pw,
        "_userMetaToken": self.ecfs_usermetatoken
    }
    self.ecfs_login_register_headers = {
        'Accept': 'application/json, text/javascript, */*',
        'Accept-Encoding': 'gzip, deflate, br, zstd',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7,ja;q=0.6',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Host': 'ecfs.scourt.go.kr',
        'Origin': 'https://ecfs.scourt.go.kr',
        'Pragma': 'no-cache',
        'Referer': 'https://ecfs.scourt.go.kr/ecf/index.jsp',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest',
        'sec-ch-ua': '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"'
    }
    self.ecfs_login_register_res = requests.post(url=self.ecfs_login_register_url, headers=self.ecfs_login_register_headers, cookies=self.ecfs_cookies, data=self.ecfs_login_register_payload)

    self.ecfs_cases_read_url = "https://ecfs.scourt.go.kr/ecf/ecf/ECF260/ECF260s01Cmd.ajax"
    self.ecfs_cases_read_payload = {
        "chkHjDay":"",
        "finSaYn": "",
        "fromDate": "",
        "toDate": "",
        "bubCd": "",
        "sysCd": "",
        "checkYn": "Y",
        "gbn": "jupDay",
        "saNo": "20240010000000",
        "dsNm": "",
        "alnCol1": "JUP_DAY_DESC",
        "alnCol2": "BUB_ABB_ASC",
        "alnCol3": "SA_NO_DESC",
        "_userMetaToken": self.ecfs_usermetatoken
    }
    self.ecfs_cases_read_headers = {
        'Accept': 'application/json, text/javascript, */*',
        'Accept-Encoding': 'gzip, deflate, br, zstd',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7,ja;q=0.6',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Host': 'ecfs.scourt.go.kr',
        'Origin': 'https://ecfs.scourt.go.kr',
        'Pragma': 'no-cache',
        'Referer': 'https://ecfs.scourt.go.kr/ecf/ecf200/ECF260.jsp',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest',
        'sec-ch-ua': '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"'
    }
    self.ecfs_read_res = requests.post(url=self.ecfs_cases_read_url, headers=self.ecfs_cases_read_headers, cookies=self.ecfs_cookies, data=self.ecfs_cases_read_payload)

    return self.ecfs_read_res

  def extract_payload_keys(self):
    # 진행중사건 페이지 이동 시 사용할 payload keys를 획득 및 분류
    ing_cases_dict = json.loads(self.ecfs_read_res.text)
    ing_cases_json = json.dumps(ing_cases_dict, ensure_ascii=False, indent=2)

    ing_keys = ing_cases_dict["data"]["result"][0]
    self.param = {}

    for key,value in ing_keys.items():
        self.param.update({key:value})

    return self.param
  
  def move_to_other_page(self):
    # 진행중 사건 페이지를 조회하며 사건정보 획득
    param = self.param
    self.page_lookup_res_list = []
    for target_page in range(int(param['_duNaviPageCnt'])):
      page_lookup_payloads = f"""alnCol3={param['_duNaviParamalnCol3']}&alnCol3=&dsNm={param['_duNaviParamdsNm']}&dsNm=&alnCol2={param['_duNaviParamalnCol2']}&alnCol2=&alnCol1={param['_duNaviParamalnCol1']}&alnCol1=&saNo={param['_duNaviParamsaNo']}&saNo=&chkHjDay={param['_duNaviParamchkHjDay']}&chkHjDay=&gbn={param['_duNaviParamgbn']}&gbn=&toDate={param['_duNaviParamtoDate']}&toDate=&_userMetaToken={param['_duNaviParam_userMetaToken']}&_userMetaToken={param['_duNaviParam_userMetaToken']}&checkYn={param['_duNaviParamcheckYn']}&checkYn=&bubCd={param['_duNaviParambubCd']}&bubCd=&finSaYn={param['_duNaviParamfinSaYn']}&finSaYn=&fromDate={param['_duNaviParamfromDate']}&fromDate=&sysCd={param['_duNaviParamsysCd']}&sysCd=&userid_for_trace={param['_duNaviParamuserid_for_trace']}&userid_for_trace=&sequence_for_trace={param['_duNaviParamsequence_for_trace']}&sequence_for_trace=&ipaddress_for_trace={param['_duNaviParamipaddress_for_trace']}&ipaddress_for_trace=&idList={param['_duNaviParamidList']}&idList=&mstUserTyp={param['_duNaviParammstUserTyp']}&mstUserTyp=&loginId={param['_duNaviParamloginId']}&loginId=&mstUserId={param['_duNaviParammstUserId']}&mstUserId=&userTyp={param['_duNaviParamuserTyp']}&userTyp=&auth_jcul={param['_duNaviParamauth_jcul']}&auth_jcul=&auth_sdcfm={param['_duNaviParamauth_sdcfm']}&auth_sdcfm=&auth_sasch={param['_duNaviParamauth_sasch']}&auth_sasch=&auth_elpay={param['_duNaviParamauth_elpay']}&auth_elpay=&jsokRelCd={param['_duNaviParamjsokRelCd']}&jsokRelCd=&jpSignStatcd={param['_duNaviParamjpSignStatcd']}&jpSignStatcd=&userId={param['_duNaviParamuserId']}&userId=&toDay={param['_duNaviParamtoDay']}&toDay=&jupSday={param['_duNaviParamjupSday']}&jupSday=&jupEday={param['_duNaviParamjupEday']}&jupEday=&sysCdType={param['_duNaviParamsysCdType']}&sysCdType=&userId_devon_interate_0={param['_duNaviParamuserId_devon_interate_0']}&userId_devon_interate_0=&devonTargetRow={target_page+1}&devonTargetRow=&_duNaviRowSize={param['_duNaviRowSize']}&_duNaviRowSize="""
      # 한글문자가 들어간 경우, utf-8로 인코딩하여 urlencodeing이 가능하도록 payload 생성
      page_lookup_payloads = page_lookup_payloads.encode("utf-8")

      page_lookup_url = "https://ecfs.scourt.go.kr/ecf/ecf/ECF260/ECF260s01Cmd.ajax"
      page_lookup_header = {
          'Accept': 'application/json, text/javascript, */*',
          'Accept-Encoding': 'gzip, deflate, br, zstd',
          'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7,ja;q=0.6',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
          'Content-Type': 'application/x-www-form-urlencoded',
          'Host': 'ecfs.scourt.go.kr',
          'Origin': 'https://ecfs.scourt.go.kr',
          'Pragma': 'no-cache',
          'Referer': 'https://ecfs.scourt.go.kr/ecf/ecf200/ECF260.jsp',
          'Sec-Fetch-Dest': 'empty',
          'Sec-Fetch-Mode': 'cors',
          'Sec-Fetch-Site': 'same-origin',
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
          'X-Requested-With': 'XMLHttpRequest',
          'sec-ch-ua': '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
          'sec-ch-ua-mobile': '?0',
          'sec-ch-ua-platform': '"Windows"'
      }


      self.page_lookup_res = requests.post(url=page_lookup_url, headers=page_lookup_header, cookies=self.ecfs_cookies, data=page_lookup_payloads)
      page_lookup_json = json.loads(self.page_lookup_res.text)
      self.page_lookup_res_list.append(page_lookup_json)

    return self.page_lookup_res_list      

  def get_my_cases(self):
    # 진행중사건 획득에 필요한 함수 순차 실행
    self.get_client_cookies()
    self.post_login_session()
    self.extract_payload_keys()
    self.move_to_other_page()

  def start(self):
    # 사건획득 오류체크하여 에러 발생 시 세션 재획득 작업 시작
    self.get_my_cases()
    errcnt = 0
    errmax = 30
    while self.page_lookup_res_list[0]['meta']['err'] == "true":
        errcnt += 1
        if errcnt > errmax:
           break
        self.get_my_cases()
class ECASE(Resource):
  
  def post(self):

    try:
      # print('requested for id / pw: ')
      # print(request)
      # print(request.get_json()['ecid'])
      # print(request.get_json()['ecpw'])

      id = request.get_json()['ecid']
      pw = request.get_json()['ecpw']
      aa = ECASELIST(id=id,pw=pw)
      aa.start()
      # print(aa.page_lookup_res_list)
      page_lookup_res_list = aa.page_lookup_res_list
      if (
        page_lookup_res_list
        and page_lookup_res_list[0].get('meta', {}).get('err') == "true"
      ):
        return jsonify({"error": json.dumps(page_lookup_res_list)})
      return jsonify({"ecaselist": page_lookup_res_list}) 
      
    except Exception as e:
      return jsonify({"error": str(e)})   

class NORITOKENS(Resource):
  '''
  settings of nori index
  {
    "nori": {
      "settings": {
        "index": {
          "routing": {
              "allocation": {
                  "include": {
                      "_tier_preference": "data_content"
                  }
              }
          },
          "number_of_shards": "1",
          "provided_name": "nori",
          "creation_date": "1675227538386",
          "analysis": {
              "analyzer": {
                  "nori_discard": {
                      "filter": [
                          "lowercase",
                          "nori_readingform"
                      ],
                      "type": "custom",
                      "tokenizer": "nori_discard"
                  },
                  "nori_mixed": {
                      "filter": [
                          "lowercase",
                          "nori_readingform"
                      ],
                      "type": "custom",
                      "tokenizer": "nori_mixed"
                  },
                  "nori_none": {
                      "filter": [
                          "lowercase",
                          "nori_readingform"
                      ],
                      "type": "custom",
                      "tokenizer": "nori_none"
                  }
              },
              "tokenizer": {
                  "nori_discard": {
                      "type": "nori_tokenizer",
                      "decompound_mode": "discard"
                  },
                  "nori_mixed": {
                      "type": "nori_tokenizer",
                      "decompound_mode": "mixed"
                  },
                  "nori_none": {
                      "type": "nori_tokenizer",
                      "decompound_mode": "none"
                  }
              }
          },
          "number_of_replicas": "0",
          "uuid": "CBWR9wE0SLCTWaNWtTcYOw",
          "version": {
              "created": "7160299"
          }
        }
      }
    }
  }
  '''
  
  def put(self):
    # print('inside noritokens put flask api...')
    # print(request)
    # print(dir(request))
    if es is None:
        response = jsonify({"detail": "elasticsearch is not configured"})
        response.status_code = 503
        return response
    try:
        settings = {
            "settings": {
                "analysis":{                
                    "analyzer": {
                        "nori_discard_plus_legal_voca": {
                            "filter": [
                                "lowercase",
                                "nori_readingform",
                                "my_posfilter"
                            ],
                            "type": "custom",
                            "tokenizer": "nori_discard_plus_legal_voca"
                        },
                        "nori_none_plus_legal_voca": {
                            "filter": [
                                "lowercase",
                                "nori_readingform",
                                "my_posfilter"
                            ],
                            "type": "custom",
                            "tokenizer": "nori_none_plus_legal_voca"
                        },
                    },
                    "tokenizer": {
                        "nori_discard_plus_legal_voca": {
                            "type": "nori_tokenizer",
                            "user_dictionary_rules": request.get_json()['user_dictionary_rules'],
                            "decompound_mode": "discard"
                        },
                        "nori_none_plus_legal_voca": {
                            "type": "nori_tokenizer",
                            "user_dictionary_rules": request.get_json()['user_dictionary_rules'],
                            "decompound_mode": "none"
                        }
                    },
                    "filter": {
                        "my_posfilter": {
                            "type": "nori_part_of_speech",
                            "stoptags": request.get_json()['stoptags']
                        }
                    }
                }
            }
        }
        # mappings = {
        #   "properties": {
        #     "body": {
        #       "type": "text",
        #       "analyzer": "nori_mixed"
        #     },
        #     "title": {
        #       "type": "text",
        #       "analyzer": "nori_mixed"
        #     }
        #   },
        # } 
  
        es.indices.close(index = 'nori')
        # res = es.indices.put_mappings(body = mappings, index = 'nori')
        res = es.indices.put_settings(body = settings, index = 'nori') # elk8
        # print(res)
        es.indices.open(index='nori')
        # print(es.indices.get(index= 'nori'))
        # return jsonify(res) # elk 7 or 8
        return jsonify(res.body) # elk8 # 
        # The jsonify() function in flask returns a flask.Response() object 
        # that already has the appropriate content-type header 'application/json' 
        # for use with json responses. 
        # Whereas, the json.dumps() method will just return an encoded string, 
        # which would require manually adding the MIME type header.
    except Exception as e:
        return jsonify({"detail": str(e)})

  def post(self):
      # print('inside noritokens post flask api...')
      # print(request)
      # print(dir(request))
      # print(request.access_route)
      # print(request.content_type)
      # print(request.headers)
      # print(request.method)
      # print(request.args)
      # print(type(request.data))
      # print(request.is_json)
      # print(request.get_data().decode('utf-8'))
      # print(request.get_json())
      # print(type(request.get_json())) # error
      # print('break point passed')
      if es is None:
          response = jsonify({"detail": "elasticsearch is not configured"})
          response.status_code = 503
          return response
      try:
          print('requested analyzer for tokenizing: ')
          print("nori_"+request.get_json()['decompound']+request.get_json()['custom'])
          body = {
                  "analyzer": "nori_"+request.get_json()['decompound']+request.get_json()['custom'], 
                  # "analyzer": "nori_"+request.get_json()['decompound']+"_plus_legal_voca", 
                  "text": request.get_json()['text'],
                  "explain": True,
                  "attributes": request.get_json()['attributes']
                  }
          res = es.indices.analyze(index = 'nori', body = body)
          # return jsonify(res) # elk 7 (client 기준)
          return jsonify(res.body) # elk 8 (client version 기준)
      except Exception as e:
          return jsonify({"detail": str(e)})

class KEYWORDS(Resource):
    
    # GET
    # kwd1 - w2v embedding에 의한 유사 키워드 검색을 위한 키워드들이 "_"로 구분된 string
    # http://localhost:5001/keywords/이행지체_이행불능  관련 검색어 추천        
    def get(self, kwd1): 
      if es is None:
        response = jsonify({"error": "elasticsearch is not configured"})
        response.status_code = 503
        return response
      # print("KEYWORDS kwd1: ", kwd1)
      tupleList = tuple(kwd1.split("_"))
      result = keywords_query(tupleList)
      
      # result type: DataFrame
      # print('keywords query result info:')
      # print(result)
      
      # DataFrame to Python Dictionary
      result_dict = result.transpose().to_dict()
      # print('keywords result to dict:', type(result_dict))
      
      # Python Dictionary to <class 'flask.wrappers.Response'>
      jsn = jsonify(result_dict)
      # print('keywords json converted: ', type(jsn))
      return jsn # dict를 json으로 변환하여 response

def keywords_query(tupleList):

    global model_wv
    # global okt
    global word2vec_wordList
    global numOfrecommended
    
    # print("inside ml backend kwyords query func, keywords for kwd1:")
    # print(tupleList)
    listFromTuple = []
    my_list = []
    for word in tupleList:
      headers_dict = {
        "Content-type": "application/json",
        "Accept":"application/json" 
        }
      body_dict = {
        "analyzer": "nori_discard_plus_legal_voca",
        "text": word,
        "attributes" : ["leftPOS", "rightPOS", "token"],
        "explain": True, 
      }
      res = es.indices.analyze(index = 'nori', body = body_dict, headers=headers_dict)
      # print('keywords query - res from es analyze, res: ', res)
      jsn =  res.body  # es client v 8.7 + es db v 8.7
      # print("keywords query analyzed by nori: ")
      # pprint(jsn)
      tokenized = []
      for xdict in jsn['detail']['tokenfilters']:
        if xdict['name'] == 'my_posfilter':
            for ydict in xdict['tokens']:
                tokenized.append(ydict['token'])

      for token in tokenized:        
        if token in word2vec_wordList:
            listFromTuple.append(token)
    # print(listFromTuple)

    if len(listFromTuple) != 0:
      result = model_wv.wv.most_similar(
          positive = listFromTuple,  
          topn=numOfrecommended
        )
      # print("result from w2v model sim:")
      # print(result)
      loop = numOfrecommended if numOfrecommended <= len(result) else len(result)
      for n in range(loop):
        my_list.append({'keyword': result[n][0], 'sim': result[n][1], 'no': n+1})
      
      return pd.DataFrame(my_list)
    else:
      return pd.DataFrame([{'keyword': '관련 키워드가 없습니다.', 'sim': 0, 'no': 1}])
    
class RELCASES(Resource):
    
    # GET
    # kwd1 - d2v embedding에 의한 유사 case_full_no 검색을 위한 case_no들이 "_"로 구분된 string
    # http://localhost:5001/relcases/98도16_2001도3959 -> 관련판례
    def get(self, kwd1): 
        # print("relcases kwd1: ", kwd1)
        v_list = kwd1.split('_') # 현재 한 번에 하나 판례를 가지고 처리하지만 확장성 고려하여 리스트로 변환
        result = relcases_query(v_list)
        
        # result type: DataFrame
        # print('relcases query result info:')
        # print(result.info())
        
        # DataFrame to Python Dictionary
        result_dict = result.transpose().to_dict()
        # print('relcases result to dict:', type(result_dict))
        
        # Python Dictionary to <class 'flask.wrappers.Response'>
        jsn = jsonify(result_dict)
        # print('relcases json converted: ', type(jsn))
        return jsn # dict를 json으로 변환하여 response

def relcases_query(case_no_List):

    global df_d2v_tagtable
    global model_dv
    global numOfrecommended
    # print(case_no_List)
    rList = []
    for case_no in case_no_List:
      df = df_d2v_tagtable[df_d2v_tagtable['case_full_no'].str.contains(case_no)] 

      if len(df) > 0:
        # 기준판례
        # print('\n')
        # print("기준판례 사건번호/판결요지 이유 사항 토큰 리스트")
        # print(case_no)
        # print()
        # print(df.iloc[0][0])
        # print()
        # print(df.iloc[0][1])
        pass
      else:
        print('there is no data in tag table for case_no: ', case_no)
        continue

      rList.append(df)
    # 유사판례 
    # 키워드 사건번호들의 각 판결이유에 가장 유사한 벡터를 모은 리스트를 이용해서 가장 유사한 판결이유를 대표하는 사건번호와 그 유사도의 튜플들의 리스트 => result 변수에 담음
    pList = []
    for df in rList:
      if len(df) > 0: # 행 개수
        pList.append(model_dv.infer_vector(df.iloc[0][1]))
      # pList.append(model_dv.infer_vector(df.iloc[0][1].split(" ")))
    # print(pList)
    # result = model_dv.docvecs.most_similar(positive = pList, topn=numOfrecommended)
    if len(pList) > 0: # 원소 개수
      # result = model_dv.docvecs.most_similar(positive = pList, topn=numOfrecommended)
      result = model_dv.dv.most_similar(positive = pList, topn=numOfrecommended)
    else:
      return pd.DataFrame([])
    # print("type of result from <model.dv.most_similar> func:") 
    # print(type(result))
    # print("유사판례 사건번호")
    # print(result)
    # https://frhyme.github.io/python-libs/gensim1_doc2vec/
    
    my_list = []
    for i in range(numOfrecommended):
      my_list.append( {'case_full_no': result[i][0]})
      # my_list.append( {'case_full_no': result[i][0], 'reasoning': df_d2v_tagtable[df_d2v_tagtable['case_full_no'].str.contains(result[i][0])].iloc[0]['reasoning']})
    # my_dict_2 = {'case_full_no': result[1][0], 'reasoning': df_d2v_tagtable[df_d2v_tagtable['case_full_no'].str.contains(result[1][0])].iloc[0]['reasoning']}
    # my_dict_3 = {'case_full_no': result[2][0], 'reasoning': df_d2v_tagtable[df_d2v_tagtable['case_full_no'].str.contains(result[2][0])].iloc[0]['reasoning']}
    # my_dict_4 = {'case_full_no': result[3][0], 'reasoning': df_d2v_tagtable[df_d2v_tagtable['case_full_no'].str.contains(result[3][0])].iloc[0]['reasoning']}
    # my_list = [my_dict_1, my_dict_2, my_dict_3, my_dict_4]
    return pd.DataFrame(my_list)

class RELCCASES(Resource):
    
    # GET
    # kwd1 - d2v embedding에 의한 유사 caseno 검색을 위한 caseno들이 "_"로 구분된 string
    # http://localhost:5001/relccases/2009헌마246_90헌마17 -> 관련판례
    def get(self, kwd1): 
        # print("relcases kwd1: ", kwd1)
        v_list = kwd1.split('_') # 현재 한 번에 하나 판례를 가지고 처리하지만 확장성 고려하여 리스트로 변환
        result = relccases_query(v_list)
        
        # result type: DataFrame
        # print('relcases query result info:')
        # print(result.info())
        
        # DataFrame to Python Dictionary
        result_dict = result.transpose().to_dict()
        # print('relcases result to dict:', type(result_dict))
        
        # Python Dictionary to <class 'flask.wrappers.Response'>
        jsn = jsonify(result_dict)
        # print('relcases json converted: ', type(jsn))
        return jsn # dict를 json으로 변환하여 response

def relccases_query(caseno_List):

    global df_d2v_ccase_tagtable
    global model_dv_ccase
    global numOfrecommended
    # print(case_full_no_List)
    rList = []
    for caseno in caseno_List:
      df = df_d2v_ccase_tagtable[df_d2v_ccase_tagtable['caseno'].str.contains(caseno)] 
      if len(df) > 0: # 행 개수
        rList.append(df)
      
      # 기준판례
      # print('\n')
      # print("기준판례 사건번호/판결요지 이유 사항 토큰 리스트")
      # print(df.iloc[0][0])
      # print()
      # print(df.iloc[0][1])

    # 유사판례 
    # 키워드 사건번호들의 각 판결이유에 가장 유사한 벡터를 모은 리스트를 이용해서 가장 유사한 판결이유를 대표하는 사건번호와 그 유사도의 튜플들의 리스트 => result 변수에 담음
    pList = []
    for df in rList:
      pList.append(model_dv_ccase.infer_vector(df.iloc[0][1]))
      # pList.append(model_dv.infer_vector(df.iloc[0][1].split(" ")))
    # print(pList)
    # result = model_dv.docvecs.most_similar(positive = pList, topn=numOfrecommended)
    
    if len(pList) > 0: # 원소 개수
      # result = model_dv_ccase.docvecs.most_similar(positive = pList, topn=numOfrecommended)
      result = model_dv_ccase.dv.most_similar(positive = pList, topn=numOfrecommended)
    else:
      return pd.DataFrame([])
    # print("type of result from <model.dv.most_similar> func:") 
    # print(type(result))
    # print("유사판례 사건번호")
    # print(result)
    # https://frhyme.github.io/python-libs/gensim1_doc2vec/
    
    my_list = []
    for i in range(numOfrecommended):
      my_list.append( {'caseno': result[i][0]})
      # my_list.append( {'case_full_no': result[i][0], 'reasoning': df_d2v_tagtable[df_d2v_tagtable['case_full_no'].str.contains(result[i][0])].iloc[0]['reasoning']})
    # my_dict_2 = {'case_full_no': result[1][0], 'reasoning': df_d2v_tagtable[df_d2v_tagtable['case_full_no'].str.contains(result[1][0])].iloc[0]['reasoning']}
    # my_dict_3 = {'case_full_no': result[2][0], 'reasoning': df_d2v_tagtable[df_d2v_tagtable['case_full_no'].str.contains(result[2][0])].iloc[0]['reasoning']}
    # my_dict_4 = {'case_full_no': result[3][0], 'reasoning': df_d2v_tagtable[df_d2v_tagtable['case_full_no'].str.contains(result[3][0])].iloc[0]['reasoning']}
    # my_list = [my_dict_1, my_dict_2, my_dict_3, my_dict_4]
    return pd.DataFrame(my_list)

class RELCASESUMMARIES(Resource):
    
    # GET
    # kwd1 - d2v embedding에 의한 유사 case full no no 검색을 위한 case full no no들이 "^^^"로 구분된 string
    # http://localhost:5001/relcasesummaries/대법원 2011. 3. 10. 선고 2010두9976 판결 _^_1 -> 관련판례쟁점
    def get(self, kwd1): 
        # print("relcases kwd1: ", kwd1)
        v_list = kwd1.strip().split('^^^') # 현재는 한 번에 하나의 query를 받고 있음 ({{cas full no}}[\s]_^_{{nubmer}} 형식)
        # .split just creates a string list, and 
        # when it hit your specified delimiter, 
        # it adds a new string to the list, 
        # which the following chars will be placed in, and so on. 
        # So if it doesn't hit the delimiter, 
        # everything will be at the first item of the array
        result = relcasesummaries_query(v_list)
        
        # result type: DataFrame
        # print('relcases query result info:')
        # print(result.info())
        
        # DataFrame to Python Dictionary
        result_dict = result.transpose().to_dict()
        # print('relcases result to dict:', type(result_dict))
        
        # Python Dictionary to <class 'flask.wrappers.Response'>
        jsn = jsonify(result_dict)
        # print('relcases json converted: ', type(jsn))
        return jsn # dict를 json으로 변환하여 response

def relcasesummaries_query(case_full_no_no_List):

    global df_d2v_csummary_tagtable
    global model_dv_csummary
    global numOfrecommended
    # print(case_full_no_no_List)
    rList = []
    # print(df_d2v_csummary_tagtable[:10])
    # print(df_d2v_csummary_tagtable.info())
    # print(df_d2v_csummary_tagtable['case_full_no_no'][:10])
    # print(type(df_d2v_csummary_tagtable['case_full_no_no']))
    # print(df_d2v_csummary_tagtable['case_full_no_no'][1])
    # print(type(df_d2v_csummary_tagtable['case_full_no_no'][1]))    
    # print(df_d2v_csummary_tagtable['case_full_no_no'].str.contains(df_d2v_csummary_tagtable['case_full_no_no'][1], regex=False))
    # print(df_d2v_csummary_tagtable['case_full_no_no'].str.contains('2011')) // 잘 작동
    for case_full_no_no in case_full_no_no_List:
      try:
        df = df_d2v_csummary_tagtable[df_d2v_csummary_tagtable['case_full_no_no'].str.contains(case_full_no_no.strip(), regex=False)] 
        # print(df_d2v_csummary_tagtable['case_full_no_no'].str.contains(case_full_no_no.strip()))
        if len(df) > 0:
          rList.append(df)
      except Exception as e:
        # 기준판례
        print('\n', e)
        print("기준판례 사건번호/판결요지 | 이유 사항 토큰 리스트")
        print(df)
        print(case_full_no_no)
        pass

    # 유사판례 
    # 키워드 사건번호들의 각 판결이유에 가장 유사한 벡터를 모은 리스트를 이용해서 가장 유사한 판결이유를 대표하는 사건번호와 그 유사도의 튜플들의 리스트 => result 변수에 담음
    pList = []
    for df in rList:
      if len(df) > 0:
        pList.append(model_dv_csummary.infer_vector(df.iloc[0][1]))
      # pList.append(model_dv.infer_vector(df.iloc[0][1].split(" ")))
    # print(pList)
    # result = model_dv.docvecs.most_similar(positive = pList, topn=numOfrecommended)
    try:
      # result = model_dv_csummary.docvecs.most_similar(positive = pList, topn=numOfrecommended)
      result = model_dv_csummary.dv.most_similar(positive = pList, topn=numOfrecommended)
    except Exception as e:
      print(e)
      result= []
    # print("type of result from <model.dv.most_similar> func:") 
    # print(type(result))
    # print("유사판례 사건번호")
    # print(result)
    # https://frhyme.github.io/python-libs/gensim1_doc2vec/
    
    my_list = []
    try: 
      for i in range(numOfrecommended):
        my_list.append( {'case_full_no_no': result[i][0]})
    except Exception as e:
       print(e)
      # my_list.append( {'case_full_no': result[i][0], 'reasoning': df_d2v_tagtable[df_d2v_tagtable['case_full_no'].str.contains(result[i][0])].iloc[0]['reasoning']})
    # my_dict_2 = {'case_full_no': result[1][0], 'reasoning': df_d2v_tagtable[df_d2v_tagtable['case_full_no'].str.contains(result[1][0])].iloc[0]['reasoning']}
    # my_dict_3 = {'case_full_no': result[2][0], 'reasoning': df_d2v_tagtable[df_d2v_tagtable['case_full_no'].str.contains(result[2][0])].iloc[0]['reasoning']}
    # my_dict_4 = {'case_full_no': result[3][0], 'reasoning': df_d2v_tagtable[df_d2v_tagtable['case_full_no'].str.contains(result[3][0])].iloc[0]['reasoning']}
    # my_list = [my_dict_1, my_dict_2, my_dict_3, my_dict_4]
    return pd.DataFrame(my_list)

class PHRASES(Resource):
    
    # GET
    # kwd1 - d2v embedding에 의한 유사 phrase 검색을 위한 phrase token들이 "_"로 구분된 string
    # Deprecated: GET http://localhost:5001/phrases/강제집행을_위한_집행권원은
    # Current: POST http://localhost:5001/recommendedphrases {"input": "강제집행을 위한 집행권원은"}
    def get(self): 
        if (not phrase) or (model_dvp is None) or (phraseserial is None):
            response = jsonify({"error": "recommendedphrases model is disabled or not loaded"})
            response.status_code = 503
            return response

        phrase_input = request.args.get('input')
        if not phrase_input:
            response = jsonify({"error": "missing input"})
            response.status_code = 400
            return response

        phrase_ = phrase_input.replace('_', ' ')
        result = phrases_query(phrase_)
        
        # result type: DataFrame
        # print('relcases query result info:')
        # print(result.info())
        
        # DataFrame to Python Dictionary
        result_dict = result.transpose().to_dict()
        # print('relcases result to dict:', type(result_dict))
        
        # Python Dictionary to <class 'flask.wrappers.Response'>
        jsn = jsonify(result_dict)
        # print('relcases json converted: ', type(jsn))
        return jsn # dict를 json으로 변환하여 response

    def post(self):
        if (not phrase) or (model_dvp is None) or (phraseserial is None):
            response = jsonify({"error": "recommendedphrases model is disabled or not loaded"})
            response.status_code = 503
            return response

        payload = request.get_json(silent=True) or {}
        phrase_input = payload.get('input')
        if not phrase_input:
            response = jsonify({"error": "missing input"})
            response.status_code = 400
            return response

        phrase_ = phrase_input.replace('_', ' ')
        result = phrases_query(phrase_)
        result_dict = result.transpose().to_dict()
        return jsonify(result_dict)
# 현재 모델 성능에 심각한 문제가 있어 미사용중임
def phrases_query(phrase_):

    global phraseserial
    global model_dvp
    global numOfrecommended

    try:
      body = {
              "analyzer": "nori_discard_plus_legal_voca", 
              "text": phrase_,
              "explain": True,
              "attributes": ["leftPOS", "rightPOS", "token"]
              }
      res = es.indices.analyze(index = 'nori', body = body)
      # jsn =  jsonify(res.body) # elk 7 OR 8 ????????????????????
      # jsn = jsonify(res)
      jsn = res
      tokenized = []
      for xdict in jsn['detail']['tokenfilters']:
        if xdict['name'] == 'my_posfilter':
            for ydict in xdict['tokens']:
                tokenized.append(ydict['token'])
      # print('inside backend, tokenized result:')
      # print(tokenized)
    except Exception as e:
      print(e)
      tokenized = phrase_.split()
    # print(tokenized)
    rst = model_dvp.infer_vector(tokenized)
    # print(rst)
    # result = model_dvp.docvecs.most_similar(positive = [rst], topn=numOfrecommended)
    result = model_dvp.dv.most_similar(positive = [rst], topn=numOfrecommended)
    # print(result)
    
    my_list = []
    for i in range(numOfrecommended):
      try:
        for x in phraseserial:
          if str(x[-1]) == str(result[i][0]):
            my_list.append( {'phrase': x[0]})
      except:
         break
      
    return pd.DataFrame(my_list)

class HASHTAGS(Resource):
    
  # GET
  # kwd1 - LDA에 의한 주제어 추출을 위한 string
  # http://localhost:5001/hashtags/강제집행을_위한_집행권원은 -> 관련 문구
  def get(self, input):
    
    # print('inside lda')
    # print(input)
    if type(input) != type('string') or len(input) < 100:
        return jsonify({"error": "too short or not string"})
    text = input.replace("_", ' ')
    hashtags_dict = hashtags(text)
    jsn = jsonify(hashtags_dict)
    return jsn # dict를 json으로 변환하여 response

  def post(self):
    # header = {
    #         "Content-type": "application/json",
    #         "Accept":"application/json" 
    #     }    
    # body = {
    #         "input" : text
    #     }
    # print(dir(request)) # '__annotations__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__enter__', '__eq__', '__exit__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_cached_json', '_get_file_stream', '_get_stream_for_parsing', '_load_form_data', '_parse_content_type', 'accept_charsets', 'accept_encodings', 'accept_languages', 'accept_mimetypes', 'access_control_request_headers', 'access_control_request_method', 'access_route', 'application', 'args', 'authorization', 'base_url', 'blueprint', 'blueprints', 'cache_control', 'charset', 'close', 'content_encoding', 'content_length', 'content_md5', 'content_type', 'cookies', 'data', 'date', 'dict_storage_class', 'encoding_errors', 'endpoint', 'environ', 'files', 'form', 'form_data_parser_class', 'from_values', 'full_path', 'get_data', 'get_json', 'headers', 'host', 'host_url', 'if_match', 'if_modified_since', 'if_none_match', 'if_range', 'if_unmodified_since', 'input_stream', 'is_json', 'is_multiprocess', 'is_multithread', 'is_run_once', 'is_secure', 'json', 'json_module', 'list_storage_class', 'make_form_data_parser', 'max_content_length', 'max_form_memory_size', 'max_form_parts', 'max_forwards', 'method', 'mimetype', 'mimetype_params', 'on_json_loading_failed', 'origin', 'parameter_storage_class', 'path', 'pragma', 'query_string', 'range', 'referrer', 'remote_addr', 'remote_user', 'root_path', 'root_url', 'routing_exception', 'scheme', 'script_root', 'server', 'shallow', 'stream', 'trusted_hosts', 'url', 'url_charset', 'url_root', 'url_rule', 'user_agent', 'user_agent_class', 'values', 'view_args', 'want_form_data_parsed']
    # print(request.get_json())
    input = request.get_json()['input']
    if type(input) != type('string') or len(input) < 100:
        return jsonify({"error": "too short or not string"})
    text = input.replace("_", ' ')
    hashtags_dict = hashtags(text)
    jsn = jsonify(hashtags_dict)
    return jsn # dict를 json으로 변환하여 response
  
# LDA algorithm based
def hashtags(text):

  K = 5 # 말뭉치 내에 존재하는 주제(topic)들의 갯수
  L = 5 # 주제어 묶음 속 주제어 갯수
  limit = 1000 # 문서 내 문구 갯수 제한
  repetition = 25
  alpha = 0.1
  beta = 0.1

  documents = []
  for x in text.split('.'):      
    # print("....sen....")
    # print(x)
    if x.strip() == '':
       continue
    try:
      body = {
              "analyzer": "nori_discard_plus_legal_voca", 
              "text": x,
              "explain": True,
              "attributes": ["leftPOS", "rightPOS", "token"]
              }
      res = es.indices.analyze(index = 'nori', body = body)
      # print(res)
      # print(dir(res))
      # ['__bool__', '__class__', '__class_getitem__', '__contains__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattr__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__module__', '__ne__', '__new__', '__orig_bases__', '__parameters__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__slots__', '__str__', '__subclasshook__', '__weakref__', '_body', '_is_protocol', '_meta', 
      # 'body', 'meta', 'raw']
      # print(type(res)) # <class 'elastic_transport.ObjectApiResponse'>
      # print('res.body: ')
      # print(res.body)
      # jsn =  jsonify(res.body) # elk 8 body! (client 기준)
      # jsn = jsonify(res)
      jsn = res.body
      tokenized = []
      for xdict in jsn['detail']['tokenfilters']:
        if xdict['name'] == 'my_posfilter':
            for ydict in xdict['tokens']:
                if len(ydict['token'].strip()) > 1:
                  tokenized.append(ydict['token'])
      # print('inside backend for LDA topic analysis, tokenized result for a sentence:')
      # print(tokenized)
      if len(tokenized) > 0 :
        documents.append(tokenized)
    except Exception as e:
      print(e)
      tokenized = [ token for token in x.split() ]
      if len(tokenized) > 0 :
        documents.append(tokenized)

  # print(documents[:10]) # list[list[toekn: string]]

  # a list of Counters, one for each topic
  topic_word_counts = [Counter() for _ in range(K)]

  # a list of Counters, one for each document
  document_topic_counts = [Counter() for _ in documents] 

  # a list of numbers, one for each topic
  topic_counts = [0 for _ in range(K)]

  # a list of word counts in each document
  mapped = map(len, documents)
  document_lengths = list(mapped)

  # list comprehention for in 구문의 중첩 구문은 왼쪽에서 오른쪽으로 읽어나가야 함
  distinct_words = set(word for document in documents for word in document)
  W = len(distinct_words)
  # print(f'distinct words: {W}')
  D = len(documents)
  # print(f'documents len: {D}')
  documents = documents[:limit]
  D = len(documents)
  # print(f'after applying limit, documents len: {D}')

  # initializing word to topic map in 2D matrix
  document_topics = [[random.randrange(K) for word in document] for document in documents]

  for d in range(D):
      for word, topic in zip(documents[d], document_topics[d]):
          document_topic_counts[d][topic] += 1 
          topic_word_counts[topic][word] += 1
          topic_counts[topic] += 1

  def selectTopicIndex(weights):
      """returns i with probability weights[i] / sum(weights)"""
      total = sum(weights)
      rnd = total * random.random() # uniform between 0 and total
      for i, p in enumerate(weights):
          rnd -= p # return the smallest i such that
          if rnd <= 0: 
              return i # weights[0] + ... + weights[i] >= rnd

  def topic_weight(d, word, topic):
      """given a document and a word in that document,
      return the weight for the kth topic"""
    
      return p_word_given_topic(word, topic) * p_topic_given_document(topic, d)   
      
  def p_word_given_topic(word, topic, beta=beta):
      """the fraction of words assigned to _topic_
      that equal _word_ (plus some smoothing)"""
      return ((topic_word_counts[topic][word] + beta) / (topic_counts[topic] + W * beta))

  def p_topic_given_document(topic, d, alpha=alpha):
      """the fraction of words in document _d_
      that are assigned to _topic_ (plus some smoothing)"""
      return ((document_topic_counts[d][topic] + alpha) / (document_lengths[d] + K * alpha))

  def choose_new_topic(d, word):
          
      return selectTopicIndex([topic_weight(d, word, topic) for topic in range(K)])

  count = 0
  for epoch in range(repetition): # repetition
    count += 1
    # print(count)
    for d in range(D): # each documnet
        for i, (word, topic) in enumerate(zip(documents[d],document_topics[d])):
            
            # gibbs sampling: 특정 하나의 topic assignment를 제거한 가능도 기반임
            # 결국 특정 하나의 word / topic을 제거한 후 weights 계산해야 함
            document_topic_counts[d][topic] -= 1 # 문서별 토픽 갯수
            topic_word_counts[topic][word] -= 1 # 토픽별 단어 갯수
            topic_counts[topic] -= 1 # 토픽별 카운트
            document_lengths[d] -= 1 # 문서별 단어갯수
            
            # 주제를 선택
            new_topic = choose_new_topic(d, word)
            document_topics[d][i] = new_topic
              
            # 갯수 정보들을 주제 선택을 위한 계산 전으로 복귀시킴
            document_topic_counts[d][new_topic] += 1 # 문서별 토픽 갯수
            topic_word_counts[new_topic][word] += 1 # 토픽별 단어 갯수
            topic_counts[new_topic] += 1 # 토픽별 카운트
            document_lengths[d] += 1 # 문서별 단어갯수
                  
  df = \
      pd.DataFrame(columns=['Topic' +str(j) for j in range(1, K +1)], 
                  index=['Top'+str(i) for i in range(1, L+1)])

  for k, word_counts in enumerate(topic_word_counts):
      for ix, (word, count) in enumerate(word_counts.most_common(L)): # 각 토픽별로 top 5 단어
          df.loc['Top'+str(ix+1),'Topic'+str(k+1)] = \
              word+'({})'.format(count)
          # print(word)
                
  # print(df.info())
  # print(df.to_dict())
  return df.transpose().to_dict()

class SUMMARY(Resource):
    
    def get(self, input=None): 
        if not t5:
            response = jsonify({"error": "summary model is disabled or not loaded"})
            response.status_code = 503
            return response

        if input is None:
            input = request.args.get('input')
        if not input:
            response = jsonify({"error": "missing input"})
            response.status_code = 400
            return response

        text = input.replace('_', ' ')
        summary = summary_query(text)
        if not summary:
            response = jsonify({"error": "summary model is disabled or not loaded"})
            response.status_code = 503
            return response
        
        # Python Dictionary to <class 'flask.wrappers.Response'>
        jsn = jsonify( {"summary": summary})
        return jsn # dict를 json으로 변환하여 response
    
    def post(self):
      if not t5:
        response = jsonify({"error": "summary model is disabled or not loaded"})
        response.status_code = 503
        return response

      # header = {
      #         "Content-type": "application/json",
      #         "Accept":"application/json" 
      #     }    
      # body = {
      #         "input" : text
      #     }
      # print(dir(request))
      # print(request.get_json())
      payload = request.get_json(silent=True) or {}
      input = payload.get('input')
      if not input:
        response = jsonify({"error": "missing input"})
        response.status_code = 400
        return response
      text = input.replace('_', ' ')
      summary = summary_query(text)
      if not summary:
        response = jsonify({"error": "summary model is disabled or not loaded"})
        response.status_code = 503
        return response
      
      # Python Dictionary to <class 'flask.wrappers.Response'>
      jsn = jsonify( {"summary": summary})
      return jsn # dict를 json으로 변환하여 response
    
# 현재 모델 성능에 심각한 문제가 있어 미사용중임
def summary_query(text):

    global tokenizer
    global model_t5
    global max_input_length

    if ('tokenizer' not in globals()) or ('model_t5' not in globals()):
      return ""
    if (tokenizer is None) or (model_t5 is None):
      return ""

    # print('\ninside get summary api:')
    # print(text)
    # print()
    inputs = ["summarize: " + text]
    inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
    output = model_t5.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=100)
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]

    return predicted_title

class CASENETWORK(Resource):
    
    # GET
    def get(self, kwd1): 
        v_ = kwd1.replace('_', ' ')
        predeccessorsDict, successorsDict = casenetwork_query(v_)
        
        # Python Dictionary to <class 'flask.wrappers.Response'>
        jsn = jsonify([predeccessorsDict, successorsDict])
        return jsn # dict를 json으로 변환하여 response

def casenetwork_query(case_full_no):

    global graphSummary
    global graphCorpus
    # global model_dv
    # global numOfrecommended
    # print(case_full_no_List)
    # rList = []
    # for case_full_no in case_full_no_List:
    #   df = df_d2v_tagtable[df_d2v_tagtable['case_full_no'].str.contains(case_full_no)] 
    #   rList.append(df)
    # print('\ninside get casenetwork api:')
    # print(case_full_no)
    # print()
    ii = case_full_no
    
    ii = ii.replace('*', '').strip()
    ii = ii.replace('★', '').strip()
    try:
      ii = ii.strip()

      if ii[-1] == ')':
        # input(f'... inside get case full no ... {ii}')
        if '(공' in ii.strip():
          ii = ii[:ii.rfind('(공')]
        elif '(폐' in ii.strip():
          ii = ii[:ii.rfind('(폐')]
        elif '(변' in ii.strip():
          ii = ii[:ii.rfind('(변')]
        elif '(헌' in ii.strip():
          ii = ii[:ii.rfind('(헌')]
        elif '(같은' in ii.strip():
          ii = ii[:ii.rfind('(같은')]
        elif '(관보' in ii.strip():
          ii = ii[:ii.rfind('(관보')]
        elif '(동지' in ii.strip():
          ii = ii[:ii.rfind('(동지')]
        elif '(항소' in ii.strip():
          ii = ii[:ii.rfind('(항소')]
        elif '판결)' in ii.strip():
          ii = ii[:ii.rfind(')')]
        else:
          # print('\nget case full no:')
          # print('...')
          # print(ii.strip())
          ii = ii.strip()[:-1]
          # print(ii.strip())
          # input('...')
      elif '(공' in ii:
        ii = ii[:ii.rfind('(공')]

      ii = ii.strip()
    except Exception as e:
      print('\nget case full no:')
      print(e)
      # input(f'{ii}')
      ii = ''
    case_full_no = ii
    # traversalGraph = traversal.dfs_tree(graphSummary, case_full_no)
    successorsDict = traversal.dfs_successors(graphSummary, case_full_no)
    predecessorsDict = traversal.dfs_successors(graphSummaryReversed, case_full_no)
    # predecessorsDict = traversal.dfs_predecessors(graphSummary, case_full_no) # 원하는 데이터가 아니었음
      
      # 기준판례
      # print('\n')
      # print("기준판례 사건번호/판결요지 이유 사항 토큰 리스트")
      # print(df.iloc[0][0])
      # print()
      # print(df.iloc[0][1])

    # 유사판례 
    # 키워드 사건번호들의 각 판결이유에 가장 유사한 벡터를 모은 리스트를 이용해서 가장 유사한 판결이유를 대표하는 사건번호와 그 유사도의 튜플들의 리스트 => result 변수에 담음
    # pList = []
    # for df in rList:
    #   pList.append(model_dv.infer_vector(df.iloc[0][1]))
      # pList.append(model_dv.infer_vector(df.iloc[0][1].split(" ")))
    # print(pList)
    # result = model_dv.docvecs.most_similar(positive = pList, topn=numOfrecommended)
    # result = model_dv.docvecs.most_similar(positive = pList, topn=numOfrecommended)
    # print("type of result from <model.dv.most_similar> func:") 
    # print(type(result))
    # print("유사판례 사건번호")
    # print(result)
    # https://frhyme.github.io/python-libs/gensim1_doc2vec/
    
    # my_list = []
    # for i in range(numOfrecommended):
    #   my_list.append( {'case_full_no': result[i][0]})
      # my_list.append( {'case_full_no': result[i][0], 'reasoning': df_d2v_tagtable[df_d2v_tagtable['case_full_no'].str.contains(result[i][0])].iloc[0]['reasoning']})
      # my_dict_2 = {'case_full_no': result[1][0], 'reasoning': df_d2v_tagtable[df_d2v_tagtable['case_full_no'].str.contains(result[1][0])].iloc[0]['reasoning']}
      # my_dict_3 = {'case_full_no': result[2][0], 'reasoning': df_d2v_tagtable[df_d2v_tagtable['case_full_no'].str.contains(result[2][0])].iloc[0]['reasoning']}
      # my_dict_4 = {'case_full_no': result[3][0], 'reasoning': df_d2v_tagtable[df_d2v_tagtable['case_full_no'].str.contains(result[3][0])].iloc[0]['reasoning']}
      # my_list = [my_dict_1, my_dict_2, my_dict_3, my_dict_4]
    return (predecessorsDict, successorsDict)
         
    # print("\'대법원 2010. 7. 15. 선고 2010도2527 판결\' Depth First Search Tree:")
    # print(traversal.dfs_tree(graphSummary, '대법원 2010. 7. 15. 선고 2010도2527 판결'))
    # print()
    # print("\'대법원 2010. 7. 15. 선고 2010도2527 판결\' Successors Tree:")
    # print(traversal.dfs_successors(graphSummary, '대법원 2010. 7. 15. 선고 2010도2527 판결'))
    # print()
    # print("\'대법원 2010. 7. 15. 선고 2010도2527 판결\' Predecessors Tree:")
    # print(traversal.dfs_predecessors(graphSummary, '대법원 2010. 7. 15. 선고 2010도2527 판결'))
    # print() 

    
print("server api setting...")
app = Flask(__name__)

cors = CORS(app, resources={
  r"/healthcheck": {"origin": "*"},
  r"/ecase": {"origin": "*"},
  r"/noritokens": {"origin": "*"},
  r"/keywords/*": {"origin": "*"},
  r"/relcases/*": {"origin": "*"},
  r"/relccases/*": {"origin": "*"},
  r"/relcasesummaries/*": {"origin": "*"},
  r"/recommendedphrases": {"origin": "*"},
  r"/hashtags": {"origin": "*"},
  r"/summary": {"origin": "*"},
  r"/casenetwork/*": {"origin": "*"},
})
# CORS(application, resources={r'*': {'origins': ['https://webisfree.com', 'http://localhost:8080']}})

app.config['JSON_AS_ASCII'] = False
app.config['JSON_SORT_KEYS'] = False
api = Api(app)

api.add_resource(HEALTHCHECK, '/healthcheck')

api.add_resource(ECASE, '/ecase')

api.add_resource(NORITOKENS, '/noritokens')
# POST http://localhost:5001/noritokens {"text": "this is an apple."}

api.add_resource(KEYWORDS, '/keywords/<string:kwd1>')
# http://localhost:5001/keywords/이행지체_이행불능  관련 검색어 추천        

################################################################
api.add_resource(RELCASES, '/relcases/<string:kwd1>')
# http://localhost:5001/relcases/2010도14607
# doc2vec 
# _case_CorpusRsSummaryItmGs_embedding_doc2vec.py
# AttributeError: 'Doc2Vec' object has no attribute 'neg_labels'

api.add_resource(RELCCASES, '/relccases/<string:kwd1>')
# http://localhost:5001/relccases/2009헌마246
# doc2vec
# _ccase_ItmGsBdyAct_embedding_doc2vec
# AttributeError: 'Doc2Vec' object has no attribute 'neg_labels'

api.add_resource(RELCASESUMMARIES, '/relcasesummaries/<string:kwd1>')
# http://localhost:5001/relcasesummaries/대법원 2013. 3. 28. 선고 2010도14607 판결 _^_1
# doc2vec
# _caseDecisionItem_SummaryItmGsActs_embedding_doc2vec.py
# AttributeError: 'Doc2Vec' object has no attribute 'neg_labels'

api.add_resource(PHRASES, '/recommendedphrases')
# POST http://localhost:5001/recommendedphrases {"input": "강제집행에 따른 부동산의 명도에 있어서"}
# doc2vec
# __case_SummaryGsPhrases_embedding_doc2vec.py
# AttributeError: 'Doc2Vec' object has no attribute 'neg_labels'
################################################################

api.add_resource(HASHTAGS, '/hashtags', '/hashtags/<string:input>', )
# GET http://localhost:5001/hashtags/this is an apple
# POST {"input": text}

api.add_resource(SUMMARY, '/summary', '/summary/<string:input>', )
# GET http://localhost:5001/summary/this is an apple
# POST {"input": text}

api.add_resource(CASENETWORK, '/casenetwork/<string:kwd1>')
# GET http://localhost:5001/casenetwork/대법원 2013. 3. 28. 선고 2010도14607 판결

print()

if __name__ == '__main__': 
  print("server booting...")
  app.run(host='0.0.0.0', port=5001, debug=False)
  # app.run(host='127.0.0.1', port=5001, debug=False)    
