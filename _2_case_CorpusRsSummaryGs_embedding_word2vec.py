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
MAX_CHAR_INPUT_FOR_NORITOKENIZER = 20000

ML_BE_HOST="http://192.168.0.171"
ML_BE_PORT=5001

class CorpusVoca:
    """An iterator that yields sentences (lists of str)."""
    def __init__(self, listOflists, cutoffrate):
        self.sentences = listOflists
        self.cutoff = int(cutoffrate*len(self.sentences))
        self.active_words = []
        self.inactive_words = []
        self.word_counts = Counter()
        for wordlist in self.sentences:
            for token in wordlist:
                if type(token) == type(('tu', 'ple')):
                    self.word_counts[token[0]] += 1
                elif type(token) == type('string'):
                    self.word_counts[token] += 1

        for word, word_count in self.word_counts.items():
            if word_count >= self.cutoff and len(word) > 1:
                self.active_words.append(word)
            else:
                self.inactive_words.append(word)
        print("inactive words: ")
        print(self.inactive_words)
        print()
        
    def __iter__(self):
        pbar = tqdm.tqdm(total=len(self.sentences), dynamic_ncols=True)
        for line in self.sentences:
            pbar.update(1)

            # assume there's one document per line, tokens separated by whitespace / line[0][-1] <= 사건번호
            if type(line[0]) == type(('tu', 'ple')):
               yield utils.simple_preprocess(' '.join([re.sub('[ ]+', ' ', x[0].strip()) for x in line if x[0] in self.active_words]).strip())
            elif type(line[0]) == type('string'):
               yield utils.simple_preprocess(' '.join([re.sub('[ ]+', ' ', x.strip()) for x in line if x in self.active_words]).strip())
        pbar.close()

class Lines2Model():

    def __init__(self, vsize, workers):
        self.vsize = vsize
        self.workers = workers
        self.num_cores = cpu_count() - 1
        # self.preproc_filepath = preproc_filepath

    def run(self, listOfDicFilePath, df, cname, field, minlength, s_no, preproc_filepath, engine, active_cutoff_rate, date_str):
        if preproc_filepath != None:
            preproc = pd.read_pickle(preproc_filepath)[:s_no]
            print(f'totally {len(preproc)} lines')
        else:            
            lines_ = list(df[field])[:s_no]
            lines__ = list(df[cname])[:s_no]
            lines = list(zip(lines_, lines__))
            print(f'totally {len(lines)} lines')

            # divided = np.array_split(lines, self.num_cores)
            # lineslist = [x.tolist() for x in divided]

            #### doing multiprocessing ####
            # lineslist = []
            # chunk = int(len(lines)/self.num_cores)
            # for i in range(self.num_cores+1):
            #     lineslist.append(lines[i*chunk:(i+1)*chunk])

            # tasks = []
            
            # for lines in lineslist:
            #     tasks.append([lines, engine])

            #### doing single processing ####
            preproc, engine_ = self.sentences_preproc([lines, engine])
            
            #### doing multiprocessing ####
            # jobs_finished = parmap.map(
            #     self.sentences_preproc, 
            #     tasks, 
            #     pm_pbar=True, 
            #     pm_processes=self.num_cores)
            
            # preproc = []
            # engine_ = ''
            # for result in jobs_finished:
            #     preproc.extend(result[0])
            #     engine_ = result[1]
            
            self.save(preproc, re.findall('[a-zA-Z]+', listOfDicFilePath)[-2], engine_, vsize, "preproc", date_str)
            #==========#
            # if not (os.path.isdir('./model/')):
            #     os.makedirs(os.path.join('./model/'))
            # stamp = re.findall('[a-zA-Z]+', listOfDicFilePath)[-2]
            # with open(f'./model/preproc_word2vec_from_{stamp}_with_{engine_}.preproc', 'wb') as f:
            #     pickle.dump(preproc, f)
            # print(f'preproc list of lists saved in the model directory...')
            #==========#
        
        preproc_ = []
        for line in preproc:
            if len(line) >= minlength:
                preproc_.append(line)
            else:
                print(line)
                continue

        mycorpusvoca = CorpusVoca(preproc_, active_cutoff_rate)
        print("Active Words Total Number and Most Weird Examples: ")
        print(len(mycorpusvoca.active_words))
        print(mycorpusvoca.active_words[-50:])
        self.save(mycorpusvoca.word_counts, re.findall('[a-zA-Z]+', listOfDicFilePath)[-2], engine, vsize, "wordcounts", date_str)
        print()

        model = self.sentences2vec(mycorpusvoca, self.vsize)
        self.save(model, re.findall('[a-zA-Z]+', listOfDicFilePath)[-2], engine, vsize, "model", date_str)
        #==========#
        # if not (os.path.isdir('./model/')):
        #     os.makedirs(os.path.join('./model/'))        
        # stamp = re.findall('[a-zA-Z]+', listOfDicFilePath)[-2]
        # with open(f'./model/embed_word2vec_from_{stamp}_{vsize}.model', 'wb') as f:
        #     pickle.dump(model, f)
        # print("# word2vec model saved in the model directory...")
        #==========#
        return (preproc, model)
        
    def post(self, host, port=443, route = '/', headers_dict={}, data_dict={}):
        res = requests.post(host+':'+str(port)+route, headers=headers_dict, data=json.dumps(data_dict))
        # print(type(res)) # class 'requests.models.Response'>
        # print(res) # <Response [200]>
        # print(res.text)
        # input('...')
        rdict = json.loads(res.text)
        return rdict
        '''
        result_dict = self.post(
                                HOST, 
                                port=5000, 
                                route='/noritokens', 
                                headers_dict=headers_dict, 
                                data_dict = body_dict
                                )
        '''

    def save(self, stored, stamp, engine, vsize, ext, date_str):
        #==========#
        if not (os.path.isdir(f'./model/{date_str}')):
            os.makedirs(os.path.join(f'./model/{date_str}/'))
        # stamp = re.findall('[a-zA-Z]+', listOfDicFilePath)[-2]
        print('saved at '+ f'./model/{date_str}/word2vec_from_'+stamp+'_with_'+engine+'_with_'+str(vsize)+'.'+ext)
        # with open(f'./model/word2vec_from_{stamp}_with_{engine_or_vsize_info}.{ext}', 'wb') as f:
        with open(f'./model/{date_str}/word2vec_from_'+stamp+'_with_'+engine+'_with_'+str(vsize)+'.'+ext, 'wb') as f:
            pickle.dump(stored, f)
        print(f'{ext} saved in the model directory...')
        #==========#

    # lines는 (sentence, 판례인용표시) 튜플의 리스트이고 sentence는 스트링    
    def sentences_preproc(self, listengine) -> tuple :
        # print(listengine)
        lines = listengine[0] # list[tuple(txt: string, cname: string)]
        engine = listengine[1]
        i = 0
        preproc = []
        okt = Okt()
        kkma = Kkma()

        print(lines[0])
        if len(lines[0])==2:
          for line, cname in tqdm.tqdm(lines, dynamic_ncols=True):
            try:
              if not line: 
                  print(f'no item at {i}')
                  continue
              if not cname: 
                  print(f'no cname at {i}')
                  continue
              # if i % 500 == 0:
              #     print("current(every 500) - " + str(i+1))
              i += 1

              nohangul = re.compile('[^ ㄱ-ㅣ가-힣]+') # 한글과 띄어쓰기를 제외한 모든 글자
              line = nohangul.sub(' ', line) # 한글과 띄어쓰기를 제외한 모든 부분을 제거

              # 형태소 분석
              if engine == 'Kkma':
                  try:
                      malist = kkma.pos(line) # (형태소, 품사) 튜블의 리스트
                      # 필요한 어구만 대상으로 하기
                      r = []
                      for word in malist:
                          # 어미/조사 등은 대상에서 제외 
                          # if not word[1] in ["Josa", "Eomi", "Punctuation"]:
                          if not word[1] in ["JKS", "JKC", "JKG", "JKO", "JKM", "JKI", "JKQ", "JX", "JC", "EPH", "EPT", "EPP", "EFN", "EFQ", "EFO", "EFA", "EFI", "EFR", "ECE", "ECD", "ECS", "ETN", "ETD", "UN", "UV", "UE", "MDT", "MDN", "NR", "NP", ""]:
                              r.append((word[0], cname))
                          # if word[1] in [""]:
                          #     r.append(word[0])
                      if i % 1000 == 0:
                          print(r)
                      preproc.append(r)
                  except Exception as e:
                      print("kkma pos function failed for: ")
                      print(line)
                      print(e)
              elif engine == 'Okt':
                  try:
                      malist = okt.pos(line, norm=True, stem=True) # (형태소, 품사) 튜블의 리스트
                      # 형태소는 표준어로 정규화하고 원형으로 변환함
                      # 필요한 어구만 대상으로 하기
                      r = []
                      for word in malist:
                          # 어미/조사 등은 대상에서 제외 
                          # if not word[1] in ["Josa", "Eomi", "Punctuation"]:
                          if not word[1] in ["Josa", "Eomi", "KoreanParticle"]:
                              r.append((word[0], cname))
                      if i % 1000 == 0:
                          print(r)
                      preproc.append(r)
                  except Exception as e:
                      print("okt pos function failed for: ")
                      print(line)
                      print(e)
              elif engine == 'Nori':
                  try:
                    r = []
                                        
                    headers_dict = {
                            "Content-type": "application/json",
                            "Accept":"application/json" 
                            }
                    body_dict = {
                        "text" : line[:MAX_CHAR_INPUT_FOR_NORITOKENIZER],
                        "decompound" : "none",
                        # "decompound" : "discard",
                        "custom" : "_plus_legal_voca",
                        "attributes" : ["leftPOS", "rightPOS", "token"],
                    }
                    result_dict = self.post(
                      ML_BE_HOST, 
                      port=ML_BE_PORT, 
                      route='/noritokens', 
                      headers_dict=headers_dict, 
                      data_dict = body_dict
                    )
                    # token list r 만들어서 preproc append
                    # 어떤 품사만을 r 리스트에 포함시켜야 하는지
                    # 원형을 찾는 작업도 해야 하는지 문제가 있음
                    # asyncio await 사용 고려
                    # print()
                    # print('line: ')
                    # print(line)
                    # print('result dict:')
                    # print(result_dict)
                    # input('...')
                    if type(result_dict)==type('string'):
                      print('result dict')
                      print(result_dict)
                      continue
                    if type(result_dict['detail'])==type('string'):
                      print('detail')
                      print(result_dict['detail'])
                      continue
                    for xdict in result_dict['detail']['tokenfilters']:
                        if type(xdict)==type('string'):
                          print('xdict')
                          print(xdict)
                          continue
                        if xdict['name'] == 'my_posfilter':
                            for ydict in xdict['tokens']:
                                # print(ydict['token'], ydict['leftPOS'], cname)
                                if type(ydict)==type('string'):
                                  print('ydict')
                                  print(ydict)
                                  continue
                                else:
                                  r.append((ydict['token'], ydict['leftPOS'], cname))
                                # r.append(ydict['token'])
                    # print()
                    # print(r)
                    # if len(r[0]) > 1:  
                    preproc.append(r) 
                    # preproc 구조:
                    # list[list[tuple(token:string, leftPOS:string, cname: string)]]
                  except Exception as e:
                    print("nori failed for: ")
                    print(line)
                    print(e)
              else:
                print("there is no tokenizer designated for extracting keywords...")
                input("please press key only after solving the problem...")
            except Exception as e:
              print(e)
              print('error at sentencese preproc in word2vec py...')
              continue
            
        elif len(lines[0])==3 :
          for line, cname, serial in tqdm.tqdm(lines, dynamic_ncols=True):
            if not line: 
                print(f'no item at {i}')
                continue
            # if i % 500 == 0:
            #     print("current(every 500) - " + str(i+1))
            i += 1

            nohangul = re.compile('[^ ㄱ-ㅣ가-힣]+') # 한글과 띄어쓰기를 제외한 모든 글자
            line = nohangul.sub(' ', line) # 한글과 띄어쓰기를 제외한 모든 부분을 제거

            # 형태소 분석
            if engine == 'Kkma':
                try:
                    malist = kkma.pos(line) # (형태소, 품사) 튜블의 리스트
                    # 필요한 어구만 대상으로 하기
                    r = []
                    for word in malist:
                        # 어미/조사 등은 대상에서 제외 
                        # if not word[1] in ["Josa", "Eomi", "Punctuation"]:
                        if not word[1] in ["JKS", "JKC", "JKG", "JKO", "JKM", "JKI", "JKQ", "JX", "JC", "EPH", "EPT", "EPP", "EFN", "EFQ", "EFO", "EFA", "EFI", "EFR", "ECE", "ECD", "ECS", "ETN", "ETD", "UN", "UV", "UE", "MDT", "MDN", "NR", "NP", ""]:
                            r.append((word[0], cname))
                        # if word[1] in [""]:
                        #     r.append(word[0])
                    if i % 1000 == 0:
                        print(r)
                    preproc.append(r)
                except Exception as e:
                    print("kkma pos function failed for: ")
                    print(line)
                    print(e)
            elif engine == 'Okt':
                try:
                    malist = okt.pos(line, norm=True, stem=True) # (형태소, 품사) 튜블의 리스트
                    # 형태소는 표준어로 정규화하고 원형으로 변환함
                    # 필요한 어구만 대상으로 하기
                    r = []
                    for word in malist:
                        # 어미/조사 등은 대상에서 제외 
                        # if not word[1] in ["Josa", "Eomi", "Punctuation"]:
                        if not word[1] in ["Josa", "Eomi", "KoreanParticle"]:
                            r.append((word[0], cname))
                    if i % 1000 == 0:
                        print(r)
                    preproc.append(r)
                except Exception as e:
                    print("okt pos function failed for: ")
                    print(line)
                    print(e)
            elif engine == 'Nori':
                try:
                    r = []
                    headers_dict = {
                            "Content-type": "application/json",
                            "Accept":"application/json" 
                            }
                    body_dict = {
                        "text" : line,
                        "decompound" : "none",
                        # "decompound" : "discard",
                        "custom" : "_plus_legal_voca",
                        "attributes" : ["leftPOS", "rightPOS", "token"],
                    }
                    result_dict = self.post(
                      ML_BE_HOST, 
                      port=ML_BE_PORT, 
                      route='/noritokens', 
                      headers_dict=headers_dict, 
                      data_dict = body_dict
                    )
                    # token list r 만들어서 preproc append
                    # 어떤 품사만을 r 리스트에 포함시켜야 하는지
                    # 원형을 찾는 작업도 해야 하는지 문제가 있음
                    # asyncio await 사용 고려
                    # print(type(result_dict))
                    for xdict in result_dict['detail']['tokenfilters']:
                        # print(xdict['name'])
                        if xdict['name'] == 'my_posfilter':
                            for ydict in xdict['tokens']:
                                r.append((ydict['token'], ydict['leftPOS'], cname, serial))
                                # r.append(ydict['token'])
                    # print()
                    # print(r)
                    preproc.append(r) 
                    # preproc 구조:
                    # list[list[tuple(token:string, leftPOS:string, cname: string)]]
                except Exception as e:
                    print("nori failed for: ")
                    print(line)
                    print(e)
            else:
                print("there is no tokenizer designated for extracting keywords...")
                input("please press key only after solving the problem...")

        print(f'totally {i} lines after executing w2v preproc func')

        return (preproc, engine)
    
    # sentences는 sentence의 리스트이고 sentence는 단어의 리스트
    def sentences2vec(self, sentences, vsize): 
        
        print(f'vsize: {vsize}')
        # 아래에서 여러 번 반복문이 실행되는 것으로 보임 iter default 5
        model = word2vec.Word2Vec(sentences=sentences, vector_size=vsize, workers=self.workers)
        # def __init__(self, sentences=None, corpus_file=None, size=100, alpha=0.025, window=5, min_count=5,
        #         max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
        #         sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
        #         trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH, compute_loss=False, callbacks=(),
        #         max_final_vocab=None):
        # word2vec_model = Word2Vec(word2vec_corpus,
        #                             size=100,
        #                             alpha=0.025,
        #                             window=5,
        #                             min_count=5,
        #                             sg=0,
        #                             negative=5)

        return model

if __name__ == '__main__':

    old_date_str = str(input("가장 최근에 진행했던 판례 업데이트 작업날짜를 적어주세요(예시:20240719 또는 adam):")) # 가장 최근에 수행한 업데이트 작업 날짜   
    date_str = str(input("판례 업데이트 작업을 진행하는 날짜를 적어주세요(예시:20240719):")) # 판례의 업데이트 작업을 시작한 날짜, 시작날짜는 모든 코드에서 통일되어야함.
    print(date_str)

    cname = 'cname'
    field = 'unit_str'
    s_no = 20000000
    minlength = 3
    # "cname": cname, 
    # "idx": idx_, 
    # "unit_str": str_buffer + " " + p, 
    # "gistno": gistno
    vsize = 512
    workers = 3
    # pproc_filepath = './model/preproc_word2vec_from_reasoning.preproc'
    preproc_filepath_list = [
        None, 
        # None
        f'.//model//{date_str}//word2vec_from_listForCaseSentenceForreasoning_with_Nori_with_512.preproc'
    ] # enumerate glob glob 순서가 일치해야 함 주의!
    engine = "Nori" # Kkma Okt Nori
    active_cutoff_rate = 0.00001 
    # 문장 10000개라면 1번만 출현한 단어는 제외 / 문장 백만개라면 100번도 출현하지 않은 단어는 제외

    # listOfDicFilePath = './dataset/listForCaseSentenceForreasoning.pickle'
    # listOfDicFilePath = './dataset/listForCaseSentenceForgists.pickle'
    for idx, listOfDicFilePath in enumerate(glob.glob(f'..//web2df//dataset//{date_str}//listForCaseSen*.pickle')):
        # print(type(listOfDicFilePath))
        print()
        print(listOfDicFilePath)
        # listOfDicFilePath = str(listOfDicFilePath)
        answer = input('skip?(yes/no)')
        if answer=='yes':
            continue

        print("# 데이터프레임 준비")
        listOfDic_for_corpus = pd.read_pickle(listOfDicFilePath)
        df_corpus = pd.DataFrame.from_records(listOfDic_for_corpus)
        print(df_corpus.info())

        print("# word2vec embedding")
        wvec = Lines2Model(vsize, workers)
        preproc, model = wvec.run(
                                listOfDicFilePath,
                                df_corpus, 
                                cname,
                                field, 
                                minlength,
                                s_no, 
                                preproc_filepath_list[idx],
                                engine,
                                active_cutoff_rate,
                                date_str
                                )

        # df_summary = pd.read_pickle('./dataset/lisfForCaseSentenceForgists.pickle')
        # print(df_summary.info())


        # with open('./model/preproc_word2vec.preproc', 'rb') as f:
        #     preproc = pickle.load(f)
        # model = wvec.sentences2vec(preproc, 256)

        # print("# word2vec retrieving")
        # with open('word2vec_256.model', 'rb') as f:
        #     model = pickle.load(f)
        # # with open('preproc_word2vec.preproc', 'rb') as f:
        #     preproc = pickle.load(f)


        print("# word2vec example")
        print(model.wv.most_similar(positive = ["채무불이행", "이행불능"], negative = ["변제"]))
        print(model.wv.most_similar(positive = ["변호사", "사기"], negative = ["무죄"]))
        print(model.wv["지체"])
        print(type(model.wv["지체"]))

        # nda = model.wv["지체"]
        # one_hot_matrix = np.zeros((100, 3), dtype=np.float32)
        # print(one_hot_matrix)
        # one_hot_matrix[:, 0] = nda
        # print(one_hot_matrix)

        print(model.vector_size)