import pickle
import pandas as pd

def legal_kwd2kwd_recomm(word2vec_model, pkeywords:list, nkeywords:list):
        return word2vec_model.wv.most_similar(positive = pkeywords, negative = nkeywords)

def legal_case2case_recomm(sen2vec_model, df_preproc_sen2vec, caseFullNo):
        df = df_preproc_sen2vec.loc[df_preproc_sen2vec['case_full_no'] == caseFullNo]
        result = sen2vec_model.dv.most_similar(positive = [sen2vec_model.infer_vector(df.iloc[0, 1].split(" "))], topn=5)
        return result

def legal_kwd2case_recomm(word2vec_model, sen2vec_model, pkeywords:list, nkeywords:list):
        return 0

if __name__ == '__main__':

        # DATA LOADING

        print("# 데이터프레임 준비")
        df_corpus = pd.read_pickle('C:/Users/hcjeo/VSCodeProjects/web2df/saved/20221021moved/df_glaw_corpus_fullest.pickle')
        print(df_corpus.info())
        df_summary = pd.read_pickle('C:/Users/hcjeo/VSCodeProjects/web2df/saved/20221021moved/df_glaw_summary_fullest.pickle')
        print(df_summary.info())

        print("# word2vec retrieving") # 판결이유로 단어 임베딩
        with open('./model/embed_word2vec.model', 'rb') as f:
                word2vec_model = pickle.load(f)
        print(word2vec_model.wv)
        print("# doc2vec retrieving") # 판결이유로 판결이유 임베딩
        with open('./model/embed_doc2vec.model', 'rb') as f:
                doc2vec_model = pickle.load(f)

        print("# sen2vec retrieving") # 판결요지로 판결요지 임베딩
        with open('./model/embed_sen2vec.model', 'rb') as f:
                sen2vec_model = pickle.load(f)
        with open('./model/preproc_sen2vec.preproc', 'rb') as f:
                preproc_sen2vec = pickle.load(f) # (사건번호, 단어 리스트) 튜플의 리스트
        df_preproc_sen2vec = pd.DataFrame(preproc_sen2vec, columns = ['case_full_no', 'gists'])
        for i in range(len(df_preproc_sen2vec)):
                df_preproc_sen2vec.iloc[i]['gists'] = ' '.join(df_preproc_sen2vec.iloc[i]['gists']) 

        # WORD2VEC
        
        print("# word2vec example")
        print(legal_kwd2kwd_recomm(word2vec_model, ["채무불이행", "이행불능"], ["변제"]))        

        # DOC2VEC
        
        # print("# doc2vec example")
        # result = doc2vec_model.dv.most_similar(positive = [doc2vec_model.infer_vector('이행불능 이행지체'.split(" "))], topn=10)
        # print(result)

        #SEN2VEC

        print("# sen2vec example")
        caseFullNo =df_preproc_sen2vec['case_full_no'][100]
        print(caseFullNo)
        print(legal_case2case_recomm(sen2vec_model, df_preproc_sen2vec, caseFullNo))
