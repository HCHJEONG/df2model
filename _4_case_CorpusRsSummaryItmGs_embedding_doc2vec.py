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
        pbar = tqdm.tqdm(total=len(self.sentenceList), dynamic_ncols=True)
        for idx, tt in enumerate(self.sentenceList):
            pbar.update(1)
            tag, text = tt
            # print(tag)
            # print(f"{idx}/{len(self.sentenceList)}")
            # print('')
            yield TaggedDocument(
                tags = ['%s' % tag], # tag는 case_full_no
                words = text # words는 한 sentence 단어들의 리스트
                ) 
            pbar.close()

if __name__ == '__main__':

    old_date_str = str(input("가장 최근에 진행했던 판례 업데이트 작업날짜를 적어주세요(예시:20240719 또는 adam):")) # 가장 최근에 수행한 업데이트 작업 날짜   
    date_str = str(input("판례 업데이트 작업을 진행하는 날짜를 적어주세요(예시:20240719):")) # 판례의 업데이트 작업을 시작한 날짜, 시작날짜는 모든 코드에서 통일되어야함.
    
    s_no = 200000000
    randomNo = 6789

    vsize = 512
    workers = 3
    w2v = W2V.Lines2Model(vsize, workers)

    num_cores = cpu_count() - 4
    
    fields = ['decision_items', 'decision_gists', 'reasoning']
    cname = 'case_full_no'
    engine = 'Nori'
    source = 'CorpusIGR'

    modelfilepath = f'.//model//{date_str}//doc2vec_from_'+source+'_with_'+engine+'_with_'+str(vsize)+'.model'
    sentencesfilepath = f'.//model//{date_str}//doc2vec_from_'+source+'_with_'+engine+'_with_'+str(vsize)+'.preproc'

    preproc = True
    embedding = True
    
    def save(stored, source, engine, vsize, ext, date_str):
      #==========#
      if not (os.path.isdir(f'.//model//{date_str}//')):
          os.makedirs(os.path.join(f'.//model//{date_str}//'))
      # stamp = re.findall('[a-zA-Z]+', listOfDicFilePath)[-2]
      print('saved at '+ f'.//model//{date_str}//doc2vec_from_'+source+'_with_'+engine+'_with_'+str(vsize)+'.'+ext)
      # with open(f'./model/word2vec_from_{stamp}_with_{engine_or_vsize_info}.{ext}', 'wb') as f:
      with open(f'.//model//{date_str}//doc2vec_from_'+source+'_with_'+engine+'_with_'+str(vsize)+'.'+ext, 'wb') as f:
          pickle.dump(stored, f)
      print(f'{ext} saved in the model directory...')
      #==========#

    if embedding == True:
        
        if preproc == True:

            print("# 데이터프레임 준비")
            # filename = '../web2df/saved/df_glaw_corpus.pickle'
            # with open(filename, 'rb') as f:
            #     df_corpus = pickle.load(f)
            df_corpus = pd.read_pickle(f'..//web2df//saved//{date_str}//df_glaw_corpus//df_glaw_corpus_fullest_gmeta_lmeta.pickle')
            print(df_corpus.info())
            # df_corpus = df_corpus.head(30)

            ####################
            # df_corpus = pd.read_pickle('../web2df/dataset/listForCasePhraseForgists.pickle')
            # field = 'uni_str'
            # cname = 'cname'
            ####################

            print()

            # 판결이유,요지,사항 등을 전처리하여 저장함
            print("# doc2vec prepreproc")
            lines = [] # list[tuple(txt: string, cname: string)]
            with tqdm.tqdm(total = len(df_corpus), dynamic_ncols=True) as pbar:
                for idx, itm in df_corpus.iterrows():
                    txt = ''
                    for fld in fields:
                        if type(itm[fld]) == type('string'):
                            txt = txt + ' ' + itm[fld]
                    lines.append((txt, itm[cname]))
                    pbar.update(1)
            lines = lines[:s_no]
            print(f'totally {len(lines)} lines after concatenating item/gist/reasoning')
            # for x in lines:
            #     print(x[1])
            #     print(x[0][:444])
            #     y = input('press space and enter for break...')
            #     if y == ' ':
            #         break

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
            input("\n멀티프로세싱으로 토큰나이징 전처리하려면 아무 키나 누르세요...\n")

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
            save(prepreproc, source, engine_, vsize, "prepreproc",date_str) 
            # (preprec: list[list[(token:string, leftPOS: string ,cname: string)]], engine: string)
            print()

            pbar = tqdm.tqdm(total=len(prepreproc), dynamic_ncols=True) 
            # prepreproc 구조:
            # list[list[tuple(token:string, leftPOS:string, cname: string)]]
            sentences = [] # list[tuple(cname: string, list[token: string])]
            for idx, line in enumerate(prepreproc):
                pbar.update(1)
                try:
                    # assume there's one document per line, tokens separated by whitespace / line <= [(token, pos, cname),...] line[0][-1] <= 사건번호
                    # line 구조:
                    # list[tuple(token:string, leftPOS:string, cname: string)]
                    if type(line[0]) == type(('tu', 'ple')):
                        sentences.append((line[0][-1], [x[0] for x in line]))
                    elif type(line[0]) == type('string'):
                        sentences.append((line[0][-1], [x[0] for x in line]))
                except Exception as e:
                    print(e)
                    print(f'\nException at: {idx} of prepreproc...\n')
            pbar.close()

            print("# doc2vec preproc storing")
            save(sentences, source, engine_, vsize, "preproc", date_str)

        ####################
        # w2v.save(sentences, 'notWtoVbutDOCTOVECfromPhraseGists_notWtoVbutDOCTOVECfromPhraseGists_notWtoVbutDOCTOVECfromPhraseGists', engine_, "preproc")
        ####################

        # text_data = df_corpus[['case_full_no','reasoning']][:s_no]
        # # # text_data = df_summary[['case_full_no','gists']][:s_no]
        # preproc = w2v.sentences_preproc(text_data)
        # # with open('word2vec.preproc', 'rb') as f:
        # #     preproc = pickle.load(f) # 단어의 리스트의 리스트
        # with open('./model/preproc_doc2vec.preproc', 'wb') as f:
        #     pickle.dump(preproc, f)

        # 

    print("# doc2vec embedding")
    with open(f'.//model//{date_str}//doc2vec_from_'+source+'_with_'+engine+'_with_'+str(vsize)+'.'+"preproc", 'rb') as f:
        sentences = pickle.load(f) # [(case full no , [tokens])]
    
    doc2vec_corpus = Doc2VecCorpus(sentences) # [(사건명, [단어들])] -> TaggedDocument 객체
    doc2vec_model = Doc2Vec(doc2vec_corpus) # TaggedDocument 객체 (yield) -> doc2vec model 

    print("# doc2vec model storing")
    save(doc2vec_model, source, engine, vsize, "model", date_str)
        
    # precproc model loading
    input('Do you want to load model and preproc? Then press any key...')
    with open(modelfilepath, 'rb') as f:
        doc2vec_model = pickle.load(f)
    with open(sentencesfilepath, 'rb') as f:
        sentences = pickle.load(f)
    # print(sentences[:10])

    print("# doc2vec example")
    print(sentences[randomNo][0])
    print(sentences[randomNo][1])
    # result = doc2vec_model.docvecs.most_similar(positive = [doc2vec_model.infer_vector(sentences[randomNo][1])], topn=3)
    result = doc2vec_model.dv.most_similar(positive = [doc2vec_model.infer_vector(sentences[randomNo][1])], topn=3)
    print('\nresult: ')
    print(result)

    # doc2vec_model.docvecs.most_similar('영화', topn=10)
    # doc2vec_model.docvecs.most_similar('2019도97')
    # for idx, doctag in sorted(doc2vec_model.docvecs.doctags.items(), key=lambda x:x[1].offset):
    #     print(idx, doctag)
    # https://lovit.github.io/nlp/representation/2018/03/26/word_doc_embedding/