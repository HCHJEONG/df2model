# items gists reasoning 임베딩 for 판례 to (판결요지, embedding, 판례인용부호) 추천

from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from multiprocessing import cpu_count

import pandas as pd
import pickle
import parmap
import tqdm
import os

import _2_case_CorpusRsSummaryGs_embedding_word2vec as W2V

class Doc2VecCorpus:
  def __init__(self, preproc): 
      self.sentenceList = preproc # sentenceList는 (사건명, 단어 리스트) 열의 list
  def __iter__(self):
      for tag, text in self.sentenceList:
          yield TaggedDocument(
              tags = ['%s' % tag], # tag는 case_full_no
              words = text # words는 한 sentence 단어들의 리스트
              ) 

def save(stored, source, engine, vsize, ext):
  if not (os.path.isdir('./model/')):
      os.makedirs(os.path.join('./model/'))
  print('saved at '+ './model/doc2vec_from_'+source+'_with_'+engine+'_with_'+str(vsize)+'.'+ext)
  with open('./model/doc2vec_from_'+source+'_with_'+engine+'_with_'+str(vsize)+'.'+ext, 'wb') as f:
      pickle.dump(stored, f)
  print(f'{ext} saved in the model directory...')
  
if __name__ == '__main__':

    s_no = 200000000
    randomNo = 6789

    vsize = 512
    workers = 3
    w2v = W2V.Lines2Model(vsize, workers)

    num_cores = cpu_count() - 4
    
    fields = ['unit_str']
    cname = 'cname'
    engine = 'Nori'
    source = 'SummaryGsPhrases'

    modelfilepath = './model/doc2vec_from_'+source+'_with_'+engine+'_with_'+str(vsize)+'.model'
    phrasesfilepath = './model/doc2vec_from_'+source+'_with_'+engine+'_with_'+str(vsize)+'.preproc'
    phraseserialfilepath = './model/doc2vec_from_'+source+'_with_'+engine+'_with_'+str(vsize)+'.phraseserial' 

    embedding = False
    
    if embedding == True:
        
        print("# 데이터프레임 준비")
        listOfDicsForPhrases = pd.read_pickle('../web2df/dataset/listForCasePhraseForgists.pickle')
        print(listOfDicsForPhrases[:4])

        print()

        print("# doc2vec prepreproc")
        lines = [] # list[tuple(txt: string, cname: string, serial: number)]
        with tqdm.tqdm(total = len(listOfDicsForPhrases)) as pbar:
            for serial, itm in enumerate(listOfDicsForPhrases):
                txt = ''
                if type(itm['unit_str']) == type('string'):
                    txt = txt + itm['unit_str']
                lines.append((txt, itm[cname], serial))
                pbar.update(1)
        lines = lines[:s_no] # list[tuple(phrase: string, cname: string, serial: number)]
        print(f'totally {len(lines)} lines')
        print("# doc2vec phrasesWithSerial storing")
        save(lines, source, engine, vsize, "phraseserial") 
        print()

        lineslist = []
        chunk = int(len(lines)/num_cores)
        for i in range(num_cores+1):
            lineslist.append(lines[i*chunk:(i+1)*chunk])

        tasks = []
        for lines in lineslist:
            tasks.append([lines, engine])

        '''
        ### doing single processing ####
        preproc, engine_ = self.sentences_preproc([lines, engine])

        ### multiprocessing ####
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        procs = []
        for idx, li in enumerate(tasks):
            proc = Process(target=w2v.sentences_preproc, args=(li, return_dict))
            procs.append(proc)
            proc.start()
            
        for proc in procs:
            proc.join()

        preproc = return_dict.values() # ??????????????????????????????????????
        '''

        #### doing multithreading ####
        # input("\n멀티프로세싱으로 토큰나이징 전처리하려면 아무 키나 누르세요...\n")

        jobs_finished = parmap.map(
            w2v.sentences_preproc, 
            tasks, 
            pm_pbar=True, 
            pm_processes=num_cores)
        
        prepreproc = []
        engine_ = ''
        for result in jobs_finished:
            prepreproc.extend(result[0])
            engine_ = result[1]

        print("# doc2vec prepreproc storing")
        save(prepreproc, source, engine_, vsize, "prepreproc") 
        # prepreproc: list[list[(token:string, leftPOS: string ,cname: string)]] /// engine: string
        print()

        pbar = tqdm.tqdm(total=len(prepreproc)) 
        # prepreproc 구조:
        # list[list[tuple(token:string, leftPOS:string, cname: string)]]
        phrases = [] # list[tuple(tag: number, list[token: string])]
        for idx, line in enumerate(prepreproc):
            pbar.update(1)
            try:
              # assume there's one document per line, tokens separated by whitespace / line <= [(token, pos, cname, serial),...] line[0][-1] <= serial
              # line 구조: 한 문구를 구성하는 각 token 정보를 담은 튜플들이 담긴 리스트
              # list[tuple(token:string, leftPOS:string, cname: string)]
              if type(line[0]) == type(('tu', 'ple')):
                  phrases.append((line[0][-1], [x[0] for x in line]))
              elif type(line[0]) == type('string'):
                  phrases.append((line[0][-1], [x[0] for x in line]))
            except Exception as e:
              print(e)
              print(f'\nException at: {idx} of prepreproc...\n')
        pbar.close()

        print("# doc2vec preproc storing")
        save(phrases, source, engine_, vsize, "preproc") 
        # phrases => list[tuple(tag(문구 각각의 고유 넘버): number, list[token: string])]

        print("# doc2vec embedding")
        doc2vec_pcorpus = Doc2VecCorpus(phrases) # [(각 문구의 고유번호, [단어들])] -> TaggedDocument 객체
        doc2vecp_model = Doc2Vec(doc2vec_pcorpus) # TaggedDocument 객체 (yield) -> doc2vec model 

        print("# doc2vec model storing")
        save(doc2vecp_model, source, engine, vsize, "model")
        
    # precproc model loading
    input('Do you want to load model and preproc? Then press any key...')
    with open(modelfilepath, 'rb') as f:
        doc2vecp_model = pickle.load(f)
    with open(phrasesfilepath, 'rb') as f:
        phrases = pickle.load(f) # list of tuple( serial, list[token: string] )
    with open(phraseserialfilepath, 'rb') as f:
        phraseserial = pickle.load(f) # list of tuple( serial, list[token: string] )
    # print(phrases[:10])

    print("# doc2vec example")
    print(phrases[randomNo][0]) # serial == tag
    print(phrases[randomNo][1]) # list of tokens

    result = doc2vecp_model.docvecs.most_similar(positive = [doc2vecp_model.infer_vector(phrases[randomNo][1])], topn=3)
    print('\nresult: ')
    print(result)
    print()
    phrase1 = "이행지체로 인한 손해배상을 함에 있어서"
    result = doc2vecp_model.docvecs.most_similar(positive = [doc2vecp_model.infer_vector(phrase1.split())], topn=3)
    print('\nresult: ')
    print(result) # list of serial
    resultL = []
    for x in phraseserial:
        for y in result:
          if str(x[-1]) == str(y):
            resultL.append(x)
    print(f'done for {phrases[randomNo][1]}....:')  
    print(resultL)
   
    # 사용법
    # model 그리고 phraseseral list of tuples of (serial, phrase) 준비
    # 입력된 문구를 tokenized하여 list에 담아서 infer_vector
    # 그 결과는 tag의 리스트이므로 이 tag로 phraseserial 리스트에서 원래 phrase 찾음


    # doc2vec_model.docvecs.most_similar('영화', topn=10)
    # doc2vec_model.docvecs.most_similar('2019도97')
    # for idx, doctag in sorted(doc2vec_model.docvecs.doctags.items(), key=lambda x:x[1].offset):
    #     print(idx, doctag)
    # https://lovit.github.io/nlp/representation/2018/03/26/word_doc_embedding/