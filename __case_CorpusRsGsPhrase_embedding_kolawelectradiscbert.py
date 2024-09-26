import torch
from transformers import ElectraTokenizer #  gen disc 공통
from transformers import ElectraModel
from multiprocessing import cpu_count

import pandas as pd
import numpy as np
import random

import pickle
import parmap
import faiss # conda install -c pytorch/label/nightly faiss-cpu # 20230918
import tqdm
import gc
 

# columns = ['cname', 'gistno', 'idx', 'unit_str']
filepath_df1 = '..//web2df//dataset//listForCasePhraseForgists.pickle'
filepath_df2 = '..//web2df//dataset//listForCasePhraseForreasoning.pickle'
filepathlist = [
    filepath_df1,
    # filepath_df2
    ]
# savedmodelname = './/model//embedded_pharase_from_CorpusRsGs_with_kolawbert20230302.pickle'
# savedfaissindexname = './/model//faiss_embedded_pharase_from_CorpusRsGs_with_kolawbert20230302.pickle'
savedmodelname = './/model//embedded_pharase_from_CorpusGs_with_kolawbert20230302.pickle'
savedfaissindexname = './/model//faiss_embedded_pharase_from_CorpusGs_with_kolawbert20230302.pickle'

# multi = True
multi = False
# num_cores = 500000
num_cores = cpu_count() - 2
id = 0
idx = 1 # if not multi, how many chunks of data for embedding

def dynamically_embed(df_model_tokenizer):

    df = df_model_tokenizer[0]
    model = df_model_tokenizer[1]
    tokenizer = df_model_tokenizer[2]
    
    embedding_phrase_cname_tuple_list_ = []
    with tqdm.tqdm(total=len(df)) as pbar:
        for itm in df.itertuples():
            if type(itm.unit_str) != type('string') or len(itm.unit_str) < 10 :
                continue
            encoded_tokens = tokenizer.encode(
                                    itm.unit_str, 
                                    return_tensors="pt", 
                                    max_length=512,
                                    truncation=True
                                    )
            # representation = model(encoded_tokens)
            # embedding_tensor = representation.last_hidden_state[0]
            # embedding = embedding_tensor.detach().cpu().numpy().tolist()
            
            with torch.no_grad():
                representation = model(encoded_tokens)
            embedding_tensor = representation.last_hidden_state[:, 0]            
            embedding = embedding_tensor.cpu().numpy()
            
            embedding_phrase_cname_tuple_list_.append(
                (embedding, itm.unit_str, itm.cname)
                )
            # print()
            # print(embedding_phrase_cname_tuple_list_[-1])
            # print()
            pbar.update(1)
            gc.collect()

    return embedding_phrase_cname_tuple_list_


if __name__ == '__main__':

    if input('Do you want to skip embedding and go to test? (y/n) ')!='y':

        print('\ntokenizer for discriminator downloading: \n')
        tokenizer =\
            ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")    
            
        model = ElectraModel.from_pretrained('./model/koelectra_model_epch_1_btch_131899_2023-02-24-18-57-03/')
        model.eval()

        sentence = "원고가 낙찰받기 직전의 입찰명령에 첨부된 부동산목록에 피고가 임차인으로 된 임대차의 기간을 1996. 7. 30.까지로 표시하였다."
        print('\nreal sentence: ')
        print(sentence)
        real_tokens = tokenizer.tokenize(sentence)
        print('\nreal real tokens: ')
        print(real_tokens)
        real_inputs = tokenizer.encode(sentence, return_tensors="pt")
        print('\nreal inputs: ')
        print(real_inputs)
        print(real_inputs.shape)
        print()
        output = model(real_inputs)
        print('\nreal model outputs: ')
        print(output.last_hidden_state)
        print(output.last_hidden_state.shape)

        list_temp = []
        for idx, x in enumerate(filepathlist):
            print(f'\ndataset {idx} loading...')
            with open(x, 'rb') as f:
                listOfTuple = pickle.load(f)
            list_temp.append(pd.DataFrame.from_records(listOfTuple, columns = ['cname', 'gistno', 'idx', 'unit_str']))
            print(list_temp[-1].info())
        
        print('\nconcatenated dataframe: ')
        df = pd.concat(list_temp, axis=0, ignore_index=True)
        print(df.info())

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
        lineslist = []
        chunk = int(len(df)/num_cores)
        for i in range(num_cores+1):
            lineslist.append(df.iloc[i*chunk:(i+1)*chunk])

        tasks = []
        for lines in lineslist:
            tasks.append([lines, model, tokenizer])
        print('\nembedding with kolawbert model for phrases...\n')
        print(f'\nnum cores: {num_cores}\n')
        print(f'\nnum tasks: {len(tasks)}\n')

        if multi:

            jobs_finished = parmap.map(
                dynamically_embed, 
                tasks, 
                pm_pbar=True, 
                pm_processes=num_cores)

            embedding_phrase_cname_tuple_list = []
            for result in jobs_finished:
                embedding_phrase_cname_tuple_list.extend(result)
            with open(savedmodelname, 'wb') as f:
                pickle.dump(embedding_phrase_cname_tuple_list, f, pickle.HIGHEST_PROTOCOL)
        else:
            # embedding_phrase_cname_tuple_list = []
            # for df, model, tokenizer in tasks:
                embedding_phrase_cname_tuple_list = dynamically_embed([df, model, tokenizer])
                # id = id + 1
                with open(savedmodelname + str(id), 'wb') as f:
                    pickle.dump(embedding_phrase_cname_tuple_list, f, pickle.HIGHEST_PROTOCOL)
                # embedding_phrase_cname_tuple_list = []
            # for idx, _ in enumerate(tasks):
            #     with open(savedmodelname + str(idx+1), 'rb') as f:
            #         listOfTuple = pickle.load(f)
            #     embedding_phrase_cname_tuple_list = embedding_phrase_cname_tuple_list + listOfTuple

        input('\n정상 완료되었습니다... Press any key and enter key... \n')


    # 코사인 유사도 (Cosine Similarity) 를 이용해서 가장 가까운 벡터를 찾으려면 몇가지를 바꿔줘야 한다.
    # 코사인 유사도 (Cosine Similarity) 를 사용하려면 벡터 내적으로 색인하는 index를 만들면 된다.
    # 코사인 유사도를 계산하라면 벡터 내적을 필연적으로 계산해야 하기 때문이다.
    embedding_phrase_cname_tuple_list = []
    if multi:
        with open(savedmodelname, 'rb') as f:
            listOfTuple = pickle.load(f)
        embedding_phrase_cname_tuple_list = listOfTuple
    else:
        for id in range(idx):
            with open(savedmodelname + str(id), 'rb') as f:
                listOfTuple = pickle.load(f)
            embedding_phrase_cname_tuple_list = embedding_phrase_cname_tuple_list + listOfTuple

    # 생성할 때 Inner Product을 검색할 수 있는 index를 생성한다.
    index = faiss.IndexFlatIP(768)
    # 아래는 위와 동일하다.
    # index = faiss.index_factory(300, "Flat", faiss.METRIC_INNER_PRODUCT)

    # Vector를 numpy array로 바꾸기
    print('extracting vectors from embedding_phrase_cname_tuple_list...')
    vectors = []
    for em, ph, cn in tqdm.tqdm(embedding_phrase_cname_tuple_list):
        vectors.append(np.array(em[0]))
    vectors = np.array(vectors).astype(np.float32)
    # vectors를 노말라이즈 해준다.
    faiss.normalize_L2(vectors)
    index.add(vectors) 

    # query vector를 하나 만들기
    query_vector = np.array([[random.uniform(0, 1) for x in range(768)]]).astype(np.float32)
    print("query vector: {}".format(query_vector))


    # 가장 가까운 것 5개 찾기
    distances, indices = index.search(query_vector, 5)
    print('indices: ')
    print(indices)
    print('distances: ')
    print(distances)
    # 결과룰 출력하자.
    idx = 0
    for i in indices[0]:
        print('index: {}'.format(i))
        print("v{}: {}, distance={}".format(idx+1, vectors[i], distances[0][idx]))
        print(embedding_phrase_cname_tuple_list[i])
        idx += 1
        input('to continue, please press enter key...')
    # https://medium.com/analytics-vidhya/recommendation-system-using-bert-embeddings-1d8de5fc3c56
    # https://medium.com/geekculture/transformer-based-recommendation-system-b350ef9cb57
