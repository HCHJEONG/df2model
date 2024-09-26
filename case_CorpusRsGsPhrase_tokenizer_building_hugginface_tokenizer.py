from tokenizers import BertWordPieceTokenizer, SentencePieceBPETokenizer, CharBPETokenizer, ByteLevelBPETokenizer
from transformers import ElectraModel
from transformers import ElectraTokenizer #  gen disc 공통
from transformers import BertTokenizerFast

from multiprocessing import cpu_count

import os
import random
import pickle
import parmap
import faiss
import tqdm

import numpy as np
import pandas as pd

#### new tokenizer building ####
#### new tokenizer building ####
#### new tokenizer building ####
#### new tokenizer building ####
## https://keep-steady.tistory.com/37 ## mecab version
## https://monologg.kr/2020/04/27/wordpiece-vocab/#more ## pure wordpiece version
# corpus 데이터 로드
'''
from Korpora import Korpora
Korpora.corpus_list()
# Korpora.fetch('all', root_dir='./data_for_tokenizing/Korpora')
# 'kcbert', 'modu_news', 'kowikitext', 'namuwikitext'
corpus = Korpora.load('namuwikitext', root_dir='./data_for_tokenizing/Korpora')
print(type(corpus))
print(dir(corpus))
print(type(corpus.get_all_texts()))
print(dir(corpus.get_all_texts()))
for x in corpus.get_all_texts():
    print(type(x))
    print(x)

<class 'str'>
백여 년 전, Andoria 제국은 마법적인 재해로 이 세계에 불려온 The Horror 에 의해 멸망했으며, 인류는 거의 몰살되었다. 수천 ( many thousand ) 의 인간만이 바다를 건너 위험하고 개척되지 않은 섬인 Varannar 의 제국 식민지로 도망칠 수 있었다. 불신과 비난은 새 황제를 선출할 수 없게 만들었고, 네 개의 망명 왕국들이 건국되었다.
최근, 오합지졸의 왕국들은 거친 땅에서 살아남기 위해 싸우고 있으며, 종종 서로간에 전쟁을 벌이기도 하면서 서서히 쇠퇴하여 야만적으로 변하고 있다. 여러 시간이 흘러 제국과 The Horror 는 그저 옛 전설이나 동화로 여겨질 뿐이다. 하지만 신참 모험가인 당신은 그런  옛 이야기에 거의 신경쓰지 않으며, 당신의 가장 큰 관심사는 최근의 불운과 오래된 텅 빈 지갑이다.
그러나 이번만은, 행운이 당신 곁에 온 것처럼 보인다. New Garand 에서 온 한 통의 마법  편지가 당신 손안에 나타나 당신이 거대한 유산의 유일한 수령자라는 것을 말해주었다. 당 신은 Varsilia 왕국의 수도에 어떠한 친척도 기억해낼 수 없지만, 그럼에도 당신은 이같은 기회를 포기하지 않을 것이다. New Garand 로의 길은 많은 놀라운 것들을 밝혀줄 것이며,  동화와 전설이 현실로 다가올 수 있다는 것을 가르쳐줄 것이다.
<class 'str'>
전형적인 중세 판타지 세계로, Varannar 라는 큰 섬과 그 주변의 작은 섬들로 이뤄진 지역 을 배경으로 한다. 전체 지도는 120개 타일 ( 절반이상은 산, 숲, 바다, 이종족 영역 등 의미 없는 지역 ) 로 구성되어 있는데 이 중 게임 내에 구현되어 이동 가능한 타일은 총 48개다. 130개의 구역은 이 48개의 지도상의 구역과 ( 10개 구역은 도시를 포함 ) 5개의 도시  구역, 23개의 건물 구역, 53개의 던전 구역, 1개의 튜토리얼 구역으로 이루어져 있다.     
즉 총 15개의 도시가 있는데, 이 도시들은 휴식을 취하며 소문을 듣는 여관, 퀘스트를 주는 NPC, 회관 퀘스트와 창고를 제공하는 마을 회관, 상인, 각종 건물, 플레이어의 집 등으로 이루어져 있다. 이 외에 은신처와, 사람들이 모여있는 거주지나 야영지가 몇 군데 더 있으 며 이 중 일부는 퀘스트를 주거나 휴식을 취할 수 있게 해준다. 일부 던전은 여러 층으로  이루어져 있으며 대부분의 던젼은 통로나 문, 비밀통로로 나뉘어진 여러 개의 방으로 구성 되어 있다.
이외에 지도상의 구역에는 여러 적들이 배치되어 있으며 가도상에 상인이나 도적 등이 무작위로 등장하기도 한다. 플레이어는 이러한 세계에서 자유롭게 여행하게 된다.
<class 'str'>
게임 내에 등장하며 플레이어와 상호작용하는 세력들이다. 각 세력에는 평판 수치가 존재하며 왕국들의 경우 소속 도시별로도 평판 수치가 존재한다. 퀘스트, 집 구매, 건물 진입 등 에 평판이 필요한 경우가 꽤 있다.
input('...some Korpora dataset downloaded only for testing...')
'''
mecab = False

# new_vocab_size    = 35000
new_vocab_size    = 1500
limit_alphabet= 6000 # ByteLevelBPETokenizer 학습시엔 주석처리 필요
# output_path   = 'hugging_%d'%(new_vocab_size)
min_frequency = 5 # 단어의 최소 발생 빈도, 5

# 4가지중 tokenizer 선택
how_to_tokenize = BertWordPieceTokenizer  # The famous Bert tokenizer, using WordPiece
# how_to_tokenize = SentencePieceBPETokenizer  # A BPE implementation compatible with the one used by SentencePiece
# how_to_tokenize = CharBPETokenizer  # The original BPE
# how_to_tokenize = ByteLevelBPETokenizer  # The byte level version of the BPE

filepath_df1 = '..//web2df//dataset//listForCaseSentenceForgists.pickle'
filepath_df2 = '..//web2df//dataset//listForCaseSentenceForreasoning.pickle'
filepathlist = [filepath_df1, filepath_df2]
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

import pandas as pd
# f_train = pd.read_csv('data/nsmc.txt', sep='\t')
f_train = df
train_pair = [(row[0], row[3]) for _, row in tqdm.tqdm(f_train.iterrows(), total=f_train.shape[0]) if type(row[3]) == str]  # nan 제거

#  문장 및 라벨 데이터 추출
train_data  = [pair[1].replace('\n', ' ') for pair in train_pair]
train_label = [pair[0] for pair in train_pair]
print('data loading done!')
print('문장: %s' %(train_data[:3]))
print('라벨: %s' %(train_label[:3]))

# subword 학습을 위해 문장만 따로 저장
with open('./data_for_tokenizing/train_tokenizer.txt', 'w', encoding='utf-8') as f:
    for line in tqdm.tqdm(train_data):
        f.write(line+'\n')

# load korean corpus for tokenizer training
with open('./data_for_tokenizing/train_tokenizer.txt', 'r', encoding='utf-8') as f:
    data = f.read().split('\n')
print(data[:3])

num_word_list = [len(sentence.split()) for sentence in data]
print('\n코퍼스 평균/총 단어 갯수 : %.1f / %d' % (sum(num_word_list)/len(num_word_list), sum(num_word_list)))

if mecab:

    # mecab for window는 아래 코드 사용
    from konlpy.tag import Mecab  # install mecab for window: https://hong-yp-ml-records.tistory.com/91
    mecab_tokenizer = Mecab(dicpath=r"C:\mecab\mecab-ko-dic").morphs
    print('mecab check :', mecab_tokenizer('어릴때보고 지금다시봐도 재밌어요ㅋㅋ'))

    for_generation = False # or normal

    if for_generation:
        # 1: '어릴때' -> '어릴, ##때' for generation model
        total_morph=[]
        for sentence in data:
            # 문장단위 mecab 적용
            morph_sentence= []
            count = 0
            for token_mecab in mecab_tokenizer(sentence):
                token_mecab_save = token_mecab
                if count > 0:
                    token_mecab_save = "##" + token_mecab_save  # 앞에 ##를 부친다
                    morph_sentence.append(token_mecab_save)
                else:
                    morph_sentence.append(token_mecab_save)
                    count += 1
            # 문장단위 저장
            total_morph.append(morph_sentence)

    else:
        # 2: '어릴때' -> '어릴, 때'   for normal case
        total_morph=[]
        for sentence in tqdm.tqdm(data):
            # 문장단위 mecab 적용
            morph_sentence= mecab_tokenizer(sentence)
            # 문장단위 저장
            total_morph.append(morph_sentence)
                            
    print(total_morph[:3])
    print(len(total_morph))

    # mecab 적용한 데이터 저장
    # ex) 1 line: '어릴 때 보 고 지금 다시 봐도 재밌 어요 ㅋㅋ'
    with open('./data_for_tokenizing/after_mecab.txt', 'w', encoding='utf-8') as f:
        for line in tqdm.tqdm(total_morph):
            f.write(' '.join(line)+'\n')

## 1) define special tokens
user_defined_symbols = ['[BOS]','[EOS]','[UNK0]','[UNK1]','[UNK2]','[UNK3]','[UNK4]','[UNK5]','[UNK6]','[UNK7]','[UNK8]','[UNK9]']
unused_token_num = 200
unused_list = ['[unused{}]'.format(n) for n in range(unused_token_num)]
user_defined_symbols = user_defined_symbols + unused_list

print(user_defined_symbols)

# Initialize a tokenizer
if str(how_to_tokenize) == str(BertWordPieceTokenizer):
    print('BertWordPieceTokenizer')
    ## 주의!! 한국어는 strip_accents를 False로 해줘야 한다
    # 만약 True일 시 나는 -> 'ㄴ','ㅏ','ㄴ','ㅡ','ㄴ' 로 쪼개져서 처리된다
    # 학습시 False했으므로 load할 때도 False를 꼭 확인해야 한다
    new_tokenizer = BertWordPieceTokenizer(strip_accents=False,  # Must be False if cased model
                                       lowercase=False)
elif str(how_to_tokenize) == str(SentencePieceBPETokenizer):
    print('SentencePieceBPETokenizer')
    new_tokenizer = SentencePieceBPETokenizer()

elif str(how_to_tokenize) == str(CharBPETokenizer):
    print('CharBPETokenizer')
    new_tokenizer = CharBPETokenizer()
    
elif str(how_to_tokenize) == str(ByteLevelBPETokenizer):
    print('ByteLevelBPETokenizer')
    new_tokenizer = ByteLevelBPETokenizer()
       
else:
    assert('select right tokenizer')

if mecab:
    new_corpus_file   = ['./data_for_tokenizing/after_mecab.txt']  # data path
else:
    new_corpus_file   = ['./data_for_tokenizing/train_tokenizer.txt']  # data path

# Then train it!
new_tokenizer.train(files=new_corpus_file,
               vocab_size=new_vocab_size,
               min_frequency=min_frequency,  # 단어의 최소 발생 빈도, 5
               limit_alphabet=limit_alphabet,  # ByteLevelBPETokenizer 학습시엔 주석처리 필요
               show_progress=True)
print('train complete')

sentence = '나는 오늘 아침밥을 먹었다.'
output = new_tokenizer.encode(sentence)
print(sentence)
print('=>idx   : %s'%output.ids)
print('=>tokens: %s'%output.tokens)
print('=>offset: %s'%output.offsets)
print('=>decode: %s\n'%new_tokenizer.decode(output.ids))

sentence = 'I want to go my hometown'
output = new_tokenizer.encode(sentence)
print(sentence)
print('=>idx   : %s'%output.ids)
print('=>tokens: %s'%output.tokens)
print('=>offset: %s'%output.offsets)
print('=>decode: %s\n'%new_tokenizer.decode(output.ids))

# save tokenizer
hf_model_path='./data_for_tokenizing/new_tokenizer_model'
if not os.path.isdir(hf_model_path):
    os.mkdir(hf_model_path)
new_tokenizer.save_model(hf_model_path)  # vocab.txt 파일 한개가 만들어진다

tokenizer_for_load = BertTokenizerFast.from_pretrained(hf_model_path,
                                                       strip_accents=False,  # Must be False if cased model
                                                       lowercase=False)  # 로드

print('vocab size : %d' % tokenizer_for_load.vocab_size)
# tokenized_input_for_pytorch = tokenizer_for_load("i am very hungry", return_tensors="pt")
tokenized_input_for_pytorch = tokenizer_for_load("나는 오늘 아침밥을 먹었다.", return_tensors="pt")
tokenized_input_for_tensorflow = tokenizer_for_load("나는 오늘 아침밥을 먹었다.", return_tensors="tf")

print("Tokens (str)      : {}".format([tokenizer_for_load.convert_ids_to_tokens(s) for s in tokenized_input_for_pytorch['input_ids'].tolist()[0]]))
print("Tokens (int)      : {}".format(tokenized_input_for_pytorch['input_ids'].tolist()[0]))
print("Tokens (attn_mask): {}\n".format(tokenized_input_for_pytorch['attention_mask'].tolist()[0]))

# vocab check
tokenizer_for_load.get_vocab()

# special token check
tokenizer_for_load.all_special_tokens # 추가하기 전 기본적인 special token

# tokenizer에 special token 추가
special_tokens_dict = {'additional_special_tokens': user_defined_symbols}
tokenizer_for_load.add_special_tokens(special_tokens_dict)

# check tokenizer vocab with special tokens
print('check special tokens : %s'%tokenizer_for_load.all_special_tokens[:20])

# save tokenizer model with special tokens
tokenizer_for_load.save_pretrained(hf_model_path+'_special')

# check special tokens
from transformers import BertTokenizerFast
tokenizer_check = BertTokenizerFast.from_pretrained(hf_model_path+'_special')

print('check special tokens : %s'%tokenizer_check.all_special_tokens[:20])

print('vocab size : %d' % tokenizer_check.vocab_size)
tokenized_input_for_pytorch = tokenizer_check("나는 오늘 아침밥을 먹었다.", return_tensors="pt")
tokenized_input_for_tensorflow = tokenizer_check("나는 오늘 아침밥을 먹었다.", return_tensors="tf")

print("Tokens (str)      : {}".format([tokenizer_check.convert_ids_to_tokens(s) for s in tokenized_input_for_pytorch['input_ids'].tolist()[0]]))
print("Tokens (int)      : {}".format(tokenized_input_for_pytorch['input_ids'].tolist()[0]))
print("Tokens (attn_mask): {}\n".format(tokenized_input_for_pytorch['attention_mask'].tolist()[0]))

input("...new tokenizer built...")
exit()
#### new tokenizer building ####
#### new tokenizer building ####
#### new tokenizer building ####
#### new tokenizer building ####

# columns = ['cname', 'gistno', 'idx', 'unit_str']
filepath_df1 = '..//web2df//dataset//listForCasePhraseForgists.pickle'
filepath_df2 = '..//web2df//dataset//listForCasePhraseForreasoning.pickle'
filepathlist = [filepath_df1, filepath_df2]
savedmodelname = './/model//embedded_pharase_from_CorpusRsGs_with_kolawbert20230302.pickle'
savedfaissindexname = './/model//faiss_embedded_pharase_from_CorpusRsGs_with_kolawbert20230302.pickle'

# multi = True
multi = False
# num_cores = 500000
num_cores = cpu_count() - 2
id = 0

def dynamically_embed(df_model_tokenizer):

    df = df_model_tokenizer[0]
    model = df_model_tokenizer[1]
    tokenizer = df_model_tokenizer[2]
    
    embedding_phrase_cname_tuple_list_ = []
    with tqdm.tqdm(total=len(df)) as pbar:
        for itm in df.itertuples():
            tokenized = tokenizer.encode(
                                    itm.unit_str, 
                                    return_tensors="pt", 
                                    max_length=512,
                                    truncation=True
                                    )
            representation = model(tokenized)
            embedding_tensor = representation.last_hidden_state[0]
            # print(embedding_tensor)
            embedding = embedding_tensor.detach().cpu().numpy()
            # print(embedding)
            embedding_phrase_cname_tuple_list_.append(
                (embedding, itm.unit_str, itm.cname)
                )
            # print()
            # print(embedding_phrase_cname_tuple_list_[-1])
            # print()
            pbar.update(1)

    return embedding_phrase_cname_tuple_list_


if __name__ == '__main__':

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
        embedding_phrase_cname_tuple_list = []
        for df, model, tokenizer in tasks:
            embedding_phrase_cname_tuple_list = dynamically_embed([df, model, tokenizer])
            id = id + 1
            with open(savedmodelname + str(id), 'wb') as f:
                pickle.dump(embedding_phrase_cname_tuple_list, f, pickle.HIGHEST_PROTOCOL)
            embedding_phrase_cname_tuple_list = []
        for idx, _ in enumerate(tasks):
            with open(savedmodelname + str(idx+1), 'rb') as f:
                listOfTuple = pickle.load(f)
            embedding_phrase_cname_tuple_list = embedding_phrase_cname_tuple_list + listOfTuple

    input('\n정상 완료되었습니다... Press any key and enter key... \n')


    # 코사인 유사도 (Cosine Similarity) 를 이용해서 가장 가까운 벡터를 찾으려면 몇가지를 바꿔줘야 한다.
    # 코사인 유사도 (Cosine Similarity) 를 사용하려면 벡터 내적으로 색인하는 index를 만들면 된다.
    # 코사인 유사도를 계산하라면 벡터 내적을 필연적으로 계산해야 하기 때문이다.
    

    # 10차원짜리 벡터를 검색하기 위한 Faiss index를 생성
    # 생성할 때 Inner Product을 검색할 수 있는 index를 생성한다.
    index = faiss.IndexFlatIP(768)
    # 아래는 위와 동일하다.
    # index = faiss.index_factory(300, "Flat", faiss.METRIC_INNER_PRODUCT)


    # 랜덤으로 10차원 벡터를 10개 생성
    # vectors = [[random.uniform(0, 1) for _ in range(10)] for _ in range(100)]
    # Vector를 numpy array로 바꾸기
    vectors = []
    for em, ph, cn in embedding_phrase_cname_tuple_list:
        vectors.append(em[0])
    vectors = np.array(vectors).astype(np.float32)
    # vectors를 노말라이즈 해준다.
    faiss.normalize_L2(vectors)

    # 아까 만든 10x10 벡터를 Faiss index에 넣기
    index.add(vectors) 
    with open(savedfaissindexname, 'wb') as f:
        pickle.dump(embedding_phrase_cname_tuple_list, f, pickle.HIGHEST_PROTOCOL)

    # query vector를 하나 만들기
    query_vector = np.array([[random.uniform(0, 1) for x in range(768)]]).astype(np.float32)
    print("query vector: {}".format(query_vector))


    # 가장 가까운 것 5개 찾기
    distances, indices = index.search(query_vector, 5)
    # 결과룰 출력하자.
    idx = 0
    for i in indices:
        print("v{}: {}, distance={}".format(idx+1, vectors[i], distances[idx]))
        print(embedding_phrase_cname_tuple_list[i][2])
        print(embedding_phrase_cname_tuple_list[i][1])
        idx += 1
    # https://medium.com/analytics-vidhya/recommendation-system-using-bert-embeddings-1d8de5fc3c56
    # https://medium.com/geekculture/transformer-based-recommendation-system-b350ef9cb57