import re
import pickle
import pandas as pd
import numpy as np

# from collections import defaultdict  # ddict = defaultdict(list) // ddict[k] = j 
from datetime import datetime
from tqdm import tqdm

from sklearn.model_selection import train_test_split 

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from transformers import ElectraTokenizer
from transformers import ElectraForSequenceClassification
from transformers.optimization import get_cosine_schedule_with_warmup

dataFileName = './dataset/items_acts_splits.csv'
x_dataFieldName = 'items'
y_dataFieldName = 'acts_splits' # list // label column으로 가공됨

log_interval = 10000
sampleRatio = 0.03 # 실전의 경우 1

seq_length = 512 
num_labels = 220 # 분류 카테고리 수 softmax 출력 벡터 차원
# dr_rate = 0.1 # pretrained model config 기본 설정을 사용
    
batch_size = 16
epochs = 8
learning_rate = 5e-5 # 5의 일십만분의 일 # batch size 256 -> 0.1

# Scheduler 사용시 필요 
warmup_steps = None # warm up ratio에 의하여 결정됨
warmup_ratio = 0.1

# AdamW 사용시 필요
weight_decay = 0.01

# 그래디언트 정규화 하나 더
max_grad_norm = 1

padToken = 0
unkToken = 1
clsToken = 2
sepToken = 3
maskToken = 4


class Vocabulary(object):
    
    """매핑을 위해 텍스트를 처리하고 어휘 사전을 만드는 클래스 """

    def __init__(self, token_to_idx=None):
        """
        매개변수:
            token_to_idx (dict): 기존 토큰-인덱스 매핑 딕셔너리
        """

        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token 
                              for token, idx in self._token_to_idx.items()}
        
    # def to_serializable(self):
        # """ 직렬화할 수 있는 딕셔너리를 반환합니다 """
        # return {'token_to_idx': self._token_to_idx}

    # @classmethod
    # def from_serializable(cls, contents):
        # """ 직렬화된 딕셔너리에서 Vocabulary 객체를 만듭니다 """
        # return cls(**contents)

    def add_token(self, token):
        """ 토큰을 기반으로 매핑 딕셔너리를 업데이트합니다

        매개변수:
            token (str): Vocabulary에 추가할 토큰
        반환값:
            index (int): 토큰에 상응하는 정수
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index
            
    def add_many(self, tokens):
        """토큰 리스트를 Vocabulary에 추가합니다.
        
        매개변수:
            tokens (list): 문자열 토큰 리스트
        반환값:
            indices (list): 토큰 리스트에 상응되는 인덱스 리스트
        """
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        """토큰에 대응하는 인덱스를 추출합니다.
        
        매개변수:
            token (str): 찾을 토큰 
        반환값:
            index (int): 토큰에 해당하는 인덱스
        """
        return self._token_to_idx[token]

    def lookup_index(self, index):
        """ 인덱스에 해당하는 토큰을 반환합니다.
        
        매개변수: 
            index (int): 찾을 인덱스
        반환값:
            token (str): 인텍스에 해당하는 토큰
        에러:
            KeyError: 인덱스가 Vocabulary에 없을 때 발생합니다.
        """
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)

def vectorize(tokenizer, data, field, length): # -> (inputIds, atentionMask, tokenTypeIds)
    
    vectorized = tokenizer(data[field])
        
    if len(vectorized['input_ids']) >= length:
        inputIds = vectorized['input_ids'][:length-1]
        inputIds.append(3)
    else:
        inputIds = [ padToken for x in range(length)]
        inputIds[: len(vectorized['input_ids'])] = vectorized['input_ids']
        
    if len(vectorized['attention_mask']) >= length:
        attentionMask = vectorized['attention_mask'][:length]
    else:
        attentionMask = [ 0 for x in range(length)]
        attentionMask[: len(vectorized['attention_mask'])] = vectorized['attention_mask']
                
    if len(vectorized['token_type_ids']) >= length:
        tokenTypeIds = vectorized['token_type_ids'][:length]
    else:
        tokenTypeIds = [ 0 for x in range(length)]
        tokenTypeIds[: len(vectorized['token_type_ids'])] = vectorized['token_type_ids']

    # print('\n', len(inputIds), len(attentionMask), len(tokenTypeIds))
    return (inputIds, attentionMask, tokenTypeIds)

class BERTDataset(Dataset):
    
    def __init__(self, dataset, device):

        self.dataset = dataset # python dictionary type dataset
        self.device = device
        
    def __getitem__(self, index):
        """파이토치 데이터셋의 주요 진입 메서드
        
        매개변수:
            index (int): 데이터 포인트의 인덱스
        반환값:
            데이터 포인트의 특성(x_data)과 레이블(y_target) 등으로 이루어진 딕셔너리
        """
        
        input_ids = \
            torch.LongTensor(np.array(self.dataset['input_ids'][index])).to(self.device)

        attention_mask = \
            torch.LongTensor(np.array(self.dataset['attention_mask'][index])).to(self.device)
        
        token_type_ids = \
            torch.LongTensor(np.array(self.dataset['token_type_ids'][index])).to(self.device)
        
        label = \
            self.dataset['label'][index].to(self.device)
            # torch.LongTensor(np.array([self.dataset['label'][index]])).to(self.device)

        # print()
        # print("index and label: ")
        # print(index)       
        # print(label)
        # print()
        
        return [input_ids, attention_mask, token_type_ids, label]

    def __len__(self):
        return (len(self.dataset['label']))

def dataloader_factory(df, x_dataFieldName, targetDimension, device, batch_size):       
    # vectorizing
    print("1. Vectorizing...")
    data = {'input_ids':[], 'attention_mask': [], 'token_type_ids': [], 'label': []}
    
    for i in tqdm(range(len(df))):
        tuple = vectorize(tokenizer, df.iloc[i], x_dataFieldName, targetDimension)
        data['input_ids'].append(tuple[0])
        data['attention_mask'].append(tuple[1])
        data['token_type_ids'].append(tuple[2])
        data['label'].append(df.iloc[i]['label'])
        
    # datadf = pd.DataFrame(data)
    # print(datadf.info())
    # print(datadf.head())
    # print()
    
    # dataloader setting
    print("2. Torch Dataset / Torch DataLoader instantiating...")
    dataset = BERTDataset(data, device)
    dataLoader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle = True, 
                            # collate_fn=lambda x:x # 배치 리스트 요소를 데이터 개별 인스턴스로 세팅
                            )
    print("done!")
    print()
    return dataLoader 
      
#정확도 측정을 위한 함수 정의
def calc_accuracy(prediclogitsTensorList, labellogitsTensorList):
    # max_vals, max_indices = torch.max(prediclogitsTensorList, 1) #
    train_acc = np.sqrt(torch.mul((prediclogitsTensorList - labellogitsTensorList), (prediclogitsTensorList - labellogitsTensorList))\
        .sum().data.cpu().numpy()/labellogitsTensorList.size()[0])
    # 두 리스트의 같은 위치의 요소를 비교해서 조건식을 충족하는 경우에는 그 충족 횟수의 합계를 내고
    # 그 합계를 리스트의 요소 갯수로 나누어 점수를 구함
    return train_acc
    
def predict(predict_sentence):
        
    data = {'precSentences': predict_sentence, 'label': 0}
    dataloader= dataloader_factory(pd.DataFrame(data), x_dataFieldName, seq_length, 1)
    
    bertmodel.eval()

    for batch_id, item in enumerate(dataloader):
        
        input_ids, attention_mask, token_type_ids = item
        
        out = bertmodel(input_ids, attention_mask, token_type_ids)

        test_eval=[]
        for i in out.logits:
            logits=i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("민사")
            elif np.argmax(logits) == 1:
                test_eval.append("행정")
            elif np.argmax(logits) == 2:
                test_eval.append("형사")
            elif np.argmax(logits) == 3:
                test_eval.append("특허")
            elif np.argmax(logits) == 4:
                test_eval.append("가정")
            elif np.argmax(logits) == 5:
                test_eval.append("신청")
            elif np.argmax(logits) == 6:
                test_eval.append("특별")

        print(">> 입력하신 내용은 " + test_eval[0] + " 사건에 해당합니다.")
        
if __name__ == '__main__' :
    
    # CUDA 체크
    print()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA 사용여부: {}".format(torch.cuda.is_available()))
    print()
  
    # model loading
    print("model / tokenizer loading...")
    model = \
        ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator", num_labels=num_labels)
    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")        
    # model = BertForSequenceClassification.from_pretrained('skt/kobert-base-v1', num_labels=num_labels)
    with open('./model/koelectra-base-monologgv3-discriminator.pickle', 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    with open('./model/koelectra-base-monologgv3-discriminator.pickle', 'rb') as f:
        model = pickle.load(f)
    # tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    with open('./model/koelectra-base-monologgv3-discriminator_tokenizer.pickle', 'wb') as f:
        pickle.dump(tokenizer, f, pickle.HIGHEST_PROTOCOL)
    with open('./model/koelectra-base-monologgv3-discriminator_tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)
    print()
 
    # dataset loading (DataFrame)
    print("dataset loading....")
    dfFinal = pd.read_csv(dataFileName)
    print()
    
    # dataset preprocessing
    dfFinal[x_dataFieldName] = dfFinal[x_dataFieldName].\
        apply(lambda x: x.replace('lnfd', ' ') if x == x else '').\
        apply(lambda x: x.replace('【판시사항】', ' ') if x == x else '').\
        apply(lambda x: re.sub('\[.*\]', ' ', x) if x == x else '').\
        apply(lambda x: re.sub('\(.*\)', ' ', x) if x == x else '')
    dfFinal[x_dataFieldName] = dfFinal[x_dataFieldName].apply(lambda x: x.strip() if x == x else '')
    dfFinal[y_dataFieldName] = dfFinal[y_dataFieldName].\
        apply(lambda x: x.replace('구 ','') if x == x else '').\
        apply(lambda x: x.replace('시행령','') if x == x else '').\
        apply(lambda x: x.replace('시행규칙','') if x == x else '').\
        apply(lambda x: x.replace('별표','') if x == x else '').\
        apply(lambda x: x.replace('같은법시행령','') if x == x else '').\
        apply(lambda x: x.replace('같은법시행규칙','') if x == x else '').\
        apply(lambda x: x.replace('같은법','') if x == x else '').\
        apply(lambda x: re.sub('\(.*\)', '', x) if x == x else '')
    dfFinal[y_dataFieldName] = dfFinal[y_dataFieldName].apply(lambda x: x.strip() if x == x else '')
    df = dfFinal
    print(df.head(20))

    # labeling
    print("labeling...")
    vocabList = []
    labelList = []
    temp_ = df[y_dataFieldName].tolist()
    for e, i in enumerate(tqdm(temp_)):
        label_tokens = []
        try:
            tempList = eval(i)
            if len(tempList) > 0:
                for j in tempList:
                    
                    if j != '':
                        label_tokens.append(j.replace(' ', ''))
                        vocabList.append(j.replace(' ', ''))
                    else: # '[ '' ]'
                        pass
        except:
            pass
        labelList.append(list(set(label_tokens))) # 중복제거
    print('vocabList len: ', len(vocabList))
    print('labelList len: ', len(labelList))
    df = pd.concat([df, pd.Series(labelList)], axis=1, ignore_index=True)
    # print(df.info())
    # input('...')
    vocabLabelAll = Vocabulary()
    vocabLabelAll.add_many(vocabList)
    tempList = []
    for i in range(len(vocabLabelAll)):
        tempList.append(vocabLabelAll.lookup_index(i))
    print(tempList)
    print('num of applied acts exclusive of blank string: ')
    print(len(vocabLabelAll))
    print(len(list(set(vocabList))))
    print()
    
    # l = 100 if 100 < len(vocabLabel) else len(vocabLabel)
    # print(l)
    # input('...')
    # for i in range(l):
    #     print("Category Number: ")
    #     print(i)
    #     print("Case Sort Code: ")
    #     print(vocabLabel.lookup_index(i))
    #     print()
    df.columns = ['items', 'acts_splits', 'split', 'label']
    print(type(df.iloc[0]['items']))
    print(type(df.iloc[0]['acts_splits']))
    print(type(df.iloc[0]['split']))
    print(type(df.iloc[0]['label']))
    print(df.info())
    df.to_csv(dataFileName[:-4]+'_add_preproc.csv', encoding='utf-8-sig')
    
    # input('...')
    # with open('./dataset/dfFinalLabeledDf.pickle', 'wb') as f:
    #     pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)
    # with open('./dataset/dfFinalLabeledDf.pickle', 'rb') as f:
    #     df = pickle.load(f)
    
    print('num of applied acts: ')
    print(pd.Series(vocabList).value_counts())
    # print(type(df['label'].value_counts())) # pandas series
    print(pd.Series(vocabList).value_counts().index[:num_labels-1]) # unknown 하나를 위해 -1
    print(pd.Series(vocabList).value_counts().tolist()[:num_labels-1]) # unknown 하나를 위해 -1
    listOrdered = list(zip(pd.Series(vocabList).value_counts().index[:num_labels-1], pd.Series(vocabList).value_counts().tolist()[:num_labels-1]))
    print(listOrdered)
    print()
    listOrderedReverse = list(zip(pd.Series(vocabList).value_counts().index[-num_labels:], pd.Series(vocabList).value_counts().tolist()[-num_labels:]))
    print(listOrderedReverse)
    
    vocabLabel = Vocabulary()
    vocabLabel.add_many(pd.Series(vocabList).value_counts().index[:num_labels-1]) # unknown 하나를 위해 -1
    print('\nelement num of applied acts list truncated without unknown token')
    print(len(vocabLabel))
    
    # input('...')
    
    # listLabel = list(set(df['label'].tolist()))
    # dictCount = {}
    # for i in listLabel:
    #     dictCount[i] = 0
    
    # df.reset_index(inplace=True, drop=True)
    # for key, i in tqdm(df.iterrows(), total=df.shape[0]):
    #     for j in enumerate(listLabel):
    #         if df.iloc[key]['label']==j:
    #             dictCount[j] = dictCount[j] +1
    
    # listCount = sorted(dictCount.items(), reverse=True, key = lambda item: item[1])[:100]
    print("\nlabel to torch logits tensor")
    df.reset_index(inplace=True, drop=True)
    labelVecList = []
    for key, i in tqdm(df.iterrows(), total=df.shape[0]):
        
        labelVec = torch.zeros(1, num_labels, dtype=float)
        for j in i['label']:
            
            # print('\n Index of ' + j)
            
            if j in vocabLabel._token_to_idx.keys():
                # print(vocabLabel.lookup_token(j))
                labelVec = labelVec + F.one_hot(torch.tensor(vocabLabel.lookup_token(j)), num_classes=num_labels)
            else:
                # print('219')
                labelVec = labelVec + F.one_hot(torch.tensor(num_labels-1), num_classes=num_labels)
        if len(i['label']) > 0:
            labelVec = labelVec/len(i['label'])
        labelVecList.append(labelVec) 
        # print()
        # print(labelVec)
        # print()
        # input('...')
    df = pd.concat([df, pd.Series(labelVecList)], axis=1)
    df.columns = ['items', 'acts_splits', 'split', 'labelN', 'label']
    print()
    print(type(df.iloc[0]['items']))
    print(type(df.iloc[0]['acts_splits']))
    print(type(df.iloc[0]['split']))
    print(type(df.iloc[0]['labelN']))
    print(type(df.iloc[0]['label']))
    print(df.info())
    
    # with open('./dataset/dfFinalLabeledDfForKoElectra.pickle', 'wb') as f:
    #     pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)

    # print()
    # print("label truncating...")
    # df = df.drop(df[df['label'] == 2].index)
    # df = df.drop(df[df['label'] == 1].index)
    # df = df.drop(df[df['label'] == 5].index)
    # df = df.drop(df[df['label'] == 9].index)
    # df = df.drop(df[df['label'] == 3].index)
    # df = df.drop(df[df['label'] == 6].index)
    # print(df['label'].value_counts())
    # print()
    # print("label to index...")
    # df.loc[df['label']==4,  'label']= 'civil'
    # df.loc[df['label']==11, 'label']= 'admin'
    # df.loc[df['label']==12, 'label']= 'crimi'
    # df.loc[df['label']==10, 'label']= 'paten'
    # df.loc[df['label']==0,  'label']= 'famil'
    # df.loc[df['label']==8,  'label']= 'apply'
    # df.loc[df['label']==7,  'label']= 'speci'
    # print(df['label'].value_counts())
    # print()
    # df.loc[df['label']=='civil', 'label']= 0
    # df.loc[df['label']=='admin', 'label']= 1
    # df.loc[df['label']=='crimi', 'label']= 2
    # df.loc[df['label']=='paten', 'label']= 3
    # df.loc[df['label']=='famil', 'label']= 4
    # df.loc[df['label']=='apply', 'label']= 5
    # df.loc[df['label']=='speci', 'label']= 6
    # print(df['label'].value_counts())
    # print()
    
    # dataset splitting
    print("\ntrain / test datasets splitting...\n")
    # xTrain, xTest, yTrain, yTest = \
    #     train_test_split(
    #     df[x_dataFieldName], df['label'],
    #     test_size=0.2, 
    #     random_state= 42, 
    #     shuffle=True, 
    #     stratify=df['label']
    #     )
    # dfTrain = pd.concat((xTrain, yTrain), axis = 1)
    # dfTest = pd.concat((xTest, yTest), axis = 1)
    dfTrain=df.sample(frac=0.8, random_state=200) #random state is a seed value
    dfTest=df.drop(dfTrain.index)
    print("Train Dataset for Final Sampling")
    print(dfTrain.info())
    print(dfTrain.head())
    print()
    print("Testn Dataset for Final Sampling")
    print(dfTest.info())
    print(dfTest.tail())
    print()
    
    # dataset sampling
    print("train / test dataset sampling...")
    dfTrainSample = dfTrain.sample(frac=sampleRatio, random_state=999)
    dfTestSample = dfTest.sample(frac=sampleRatio, random_state=999)
    print()
    
    print("DATA VECTORIZING AND LOADING ON DATALOADER OBJECT...")
    print()
    ###########################################
    print("TRAIN DATASET...")
    print(dfTrainSample.info())
    print()
    train_loader = dataloader_factory(
                                      dfTrainSample, 
                                      x_dataFieldName, 
                                      seq_length,
                                      device,
                                      batch_size=batch_size,
                                      )    
    ###########################################
    print("TEST DATASET...")
    print(dfTestSample.info())
    print()
    test_loader = dataloader_factory(
                                     dfTestSample, 
                                     x_dataFieldName, 
                                     seq_length,
                                     device,
                                     batch_size=batch_size,
                                     )    
    ###########################################
      
    # TRAINING
    print("now training your model...")
    
    # model object setting
    bertmodel = model
    # bertmodel = model
    bertmodel.to(device)
    
    # optimizer와 scheduler 설정
    t_total = len(train_loader) * epochs
    warmup_steps = int(t_total * warmup_ratio)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bertmodel.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in bertmodel.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    scheduler = \
        get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=t_total)
        # Contant / Exponential / King / Cosine

    # loss function setting
    loss_fn = nn.CrossEntropyLoss() # 2d tensor(softmax logits) and 1d tensor(idx)|2d possibilities tensor as args
   
    # GO! 
    train_history=[]
    test_history=[]
    loss_history=[]
    
    for e in range(epochs):
        
        train_acc = 0.0
        test_acc = 0.0
        
        #TRAINING
        bertmodel.train()
        for batch_id, batch in enumerate(tqdm(train_loader)):
            
            if batch_id % log_interval == 0 : 
                print()
                print(f"Epoch : {e+1} in {epochs} / Minibatch Step : {batch_id}")

            # print(type(item))
            # print(item)
            # print(item.__dir__)
            
            # to device 처리할 단계
            input_ids, attention_mask, token_type_ids, label = batch
            label = label.squeeze(1)
            
            # print(input_ids)
            # print(attention_mask)
            # print(token_type_ids)
            # print(label.shape)
            
            # input('...')
            
            #1 gradient를 0으로 초기화
            optimizer.zero_grad()
            
            #2 출력 계산
            out = bertmodel(input_ids, 
                            attention_mask = attention_mask, 
                            token_type_ids = token_type_ids, 
                            # labels=label
                            ) 
            # print()
            # print("out:")
            # print(out)
            # print(type(out))
            # print(out.logits.shape)
            # print("label:")
            # print(label)
            # print(type(label))
            # print(label.shape)
            # input('...')
                        
            #3 손실 계산
            loss = loss_fn(out.logits, label)
            # loss = out.loss
            # print(loss)
            # input()
            
            #4 손실로 gradient 계산
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(bertmodel.parameters(), max_grad_norm)
            
            #5 계산된 gradient로 가중치를 갱신
            optimizer.step()
            
            #6 Update learning rate schedule
            scheduler.step()  
            
            #7 정확도 계산
            train_acc += calc_accuracy(out.logits, label)
            # train_acc += loss.item()
                        
            if batch_id % log_interval == 0:
                print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
                train_history.append(train_acc / (batch_id+1))
                loss_history.append(loss.data.cpu().numpy())
                
                with open(f"bert{e}model{batch_id}.model", 'wb') as f:
                    pickle.dump(bertmodel, f)
                with open(f"train{e}_history{batch_id}.history", 'wb') as f:
                    pickle.dump(train_history, f)
                with open(f"loss{e}_history{batch_id}.history", 'wb') as f:
                    pickle.dump(loss_history, f)
                
        print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
    
        # EVALUATING
        bertmodel.eval()
        for batch_id, item in enumerate(tqdm(test_loader)):
            
            if batch_id % log_interval == 0 : 
                print(f"Epoch : {e+1} in {epochs} / Minibatch Step : {batch_id}")
                        
            input_ids, attention_mask, token_type_ids, label = batch
            label = label.squeeze(1)
        
            out = bertmodel(input_ids, attention_mask, token_type_ids)
            
            test_acc += calc_accuracy(out.logits, label)

        print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
        test_history.append(test_acc / (batch_id+1))
    
    # trained model saving...
    print('trained model saving...')
    fileNameForTrainedModel = f'koelectrabasemodel_trained_{datetime.now()}.pickle'
    with open(fileNameForTrainedModel, 'wb') as f:
        pickle.dump(bertmodel, f, pickle.HIGHEST_PROTOCOL)
    print("DONE")
    
    # #질문 무한반복하기! 0 입력시 종료
    # while True :
    #     sentence = input("분류를 위한 사건 텍스트를 입력한 후 엔터키를 누르십시오: ")
    #     if sentence == "0" :
    #         break
    #     predict(sentence)
    #     print("\n")
        
