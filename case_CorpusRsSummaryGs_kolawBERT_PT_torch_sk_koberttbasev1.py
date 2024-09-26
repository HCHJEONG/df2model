import pandas as pd
import numpy as np
import pickle
import random

from tqdm import tqdm
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertForPreTraining
from transformers import XLNetTokenizer # [UNK]': 0, '[PAD]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4,  / 101 102 103 특히 103은 mask token 아님
from transformers.tokenization_utils import AddedToken
from transformers import SPIECE_UNDERLINE
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

dataFileName = ['dataset\listForCaseGistsPhrase.pickle', 'dataset\listForCaseReasoningPhrase.pickle']
x_dataFieldName = 'phrase'
groupby = 'case_full_no'
seq_length = 512 # 이는 입력을 위한 seqence length 내지 token quantity이고 최종 출력 token 하나의 임베딩 차원은 bert base에 따라 768
batch_size = 32
from_saved = False # train data from saved file or not
epochs = 16

unkToken = 0
padToken = 1
clsToken = 2
sepToken = 3
maskToken = 4

dr_rate = 0.1    

# Scheduler 사용시 필요 
warmup_steps = None
warmup_ratio = 0.1

# AdamW 사용시 필요
weight_decay = 0.01

# 그래디언트 정규화 하나 더
max_grad_norm = 1

log_interval = 10000

learning_rate = 5e-5
class KoBERTTokenizer(XLNetTokenizer):
    
    padding_side = "right"

    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        remove_space=True,
        keep_accents=False,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        additional_special_tokens=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = (
            AddedToken(mask_token, lstrip=True, rstrip=False)
            if isinstance(mask_token, str)
            else mask_token
            )

        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        super().__init__(
            vocab_file,
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            keep_accents=keep_accents,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )
        self._pad_token_type_id = 0
        self.vocab_file = vocab_file
        
    def get_vocab_file(self):
        return self.vocab_file

    def build_inputs_with_special_tokens(
            self, 
            token_ids_0: List[int], 
            token_ids_1: Optional[List[int]] = None
            ) -> List[int]:
        
        # Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        # adding special tokens. 
        # 
        # An XLNet sequence has the following format:
        # - single sequence: ``<cls> X <sep>``
        # - pair of sequences: ``<cls> A <sep> B <sep>``
        #
        # Args:
        #     token_ids_0 (:obj:`List[int]`):
        #
        #         List of IDs to which the special tokens will be added.
        #     token_ids_1 (:obj:`List[int]`, `optional`):
        #         Optional second list of IDs for sequence pairs.
        # Returns:
        #     :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def _tokenize(self, text: str) -> List[str]:
        
        # Tokenize a string
        
        text = self.preprocess_text(text)
        pieces = self.sp_model.encode(text, out_type=str, **self.sp_model_kwargs)
        new_pieces = []
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == str(",") and piece[-2].isdigit():
                cur_pieces = self.sp_model.EncodeAsPieces(
                    piece[:-1].replace(SPIECE_UNDERLINE, "")
                )
                if (
                    piece[0] != SPIECE_UNDERLINE
                    and cur_pieces[0][0] == SPIECE_UNDERLINE
                ):
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)

        return new_pieces

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], 
            token_ids_1: Optional[List[int]] = None
            ) -> List[int]:
        
        # Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        # adding special tokens. An XLNet sequence has the following format:
        # - single sequence: ``<cls> X <sep> ``
        # - pair of sequences: ``<cls> A <sep> B <sep>``
        # Args:
        #     token_ids_0 (:obj:`List[int]`):
        #         List of IDs to which the special tokens will be added.
        #     token_ids_1 (:obj:`List[int]`, `optional`):
        #         Optional second list of IDs for sequence pairs.
        # Returns:
        #     :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def create_token_type_ids_from_sequences(
                self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
            ) -> List[int]:
        
        # Create a mask from the two sequences passed to be used in a sequence-pair classification task. An XLNet
        # sequence pair mask has the following format:
        # ::
        #     0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        #     | first sequence    | second sequence |
        # If :obj:`token_ids_1` is :obj:`None`, this method only returns the first portion of the mask (0s).
        # Args:
        #     token_ids_0 (:obj:`List[int]`):
        #         List of IDs.
        #     token_ids_1 (:obj:`List[int]`, `optional`):
        #         Optional second list of IDs for sequence pairs.
        # Returns:
        #     :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
        #     sequence(s).
        
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

class NSPMLMDataset(Dataset):
    
    def __init__(self, encodings):
        self.encodings = encodings
        
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings.input_ids)
            
def dataloader_factory(df, groupby, x_dataFieldName, seq_length, device, batch_size, from_saved = False):    
       
    # vectorizing
    if from_saved == False:
            
        dfGroupedByCaseFullNo = df.groupby(groupby)
        bag = []
        text = []
        for _, item in tqdm(dfGroupedByCaseFullNo):
            # print(_)
            # print(item)
            sentences = []
            
            for _, i in item.iterrows():
                # print('..')
                # print(i['case_full_no'])
                # print(i[x_dataFieldName])
                # print(type(i))
                # print('..')
                phrase = i[x_dataFieldName].replace('판결요지', ' ').strip()
                # print(phrase)
                # input('...')
                sentences.append(phrase)
                bag.append(phrase)
            # print('one case done')
            text.append(sentences)
                
        sentence_a = []
        sentence_b = []
        label = []
        bag_size = len(bag)
        counter = 0

        for sentences in tqdm(text):
            
            num_sentences = len(sentences)
            # print(num_sentences)
            
            if num_sentences > 1:
                
                # start = random.randint(0, num_sentences-2) # 두 인자가 포함되는 범위에서 정수 발생
                start = 0
                # 50/50 whether is IsNextSentence or NotNextSentence
                if random.random() >= 0.5:
                    # print('isNext')
                    # this is IsNextSentence
                    sentence_a.append(sentences[start])
                    sentence_b.append(sentences[start+1])
                    label.append(0)
                else:
                    # print('NotNext')
                    index = random.randint(0, bag_size-1)
                    # this is NotNextSentence
                    sentence_a.append(sentences[start])
                    sentence_b.append(bag[index])
                    label.append(1)
            else:
                
                counter = counter + 1
                
        print(f'single line case: {counter}\n')
        print(len(sentence_a))
        print(len(sentence_b))
        print(len(label))
        print()

        for i in range(13):
            print(label[i])
            print(sentence_a[i] + '\n---')
            print(sentence_b[i] + '\n')
            
        # input('... press enter ...')
                    
        print("masking...")
        inputs = tokenizer(sentence_a, sentence_b, 
                            return_tensors='pt',
                            max_length=seq_length, 
                            truncation=True, 
                            padding='max_length')

        inputs['next_sentence_label'] = torch.LongTensor([label]).T # NSP label

        inputs['labels'] = inputs.input_ids.detach().clone() # MLM label

        # create random array of floats with equal dimensions to input_ids tensor
        rand = torch.rand(inputs.input_ids.shape)
        # create mask array
        mask_arr = (rand < 0.15) * (inputs.input_ids != clsToken) * \
                (inputs.input_ids != sepToken) * (inputs.input_ids != padToken)
                
        selection = []

        for i in tqdm(range(inputs.input_ids.shape[0])):
            selection.append(
                torch.flatten(mask_arr[i].nonzero()).tolist()
            )
                
        for i in tqdm(range(inputs.input_ids.shape[0])):
            inputs.input_ids[i, selection[i]] = maskToken
        
        dataset = NSPMLMDataset(inputs)    

        with open('dataset/listOfListsOfTuplesForCaseGistsPhraseDictNSP.pickle', 'wb') as f:
            pickle.dump(dataset, f)
    else:
        with open('dataset/listOfListsOfTuplesForCaseGistsPhraseDictNSP.pickle', 'rb') as f:
            dataset = pickle.load(f)

    dataLoader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True,
                            # collate_fn=lambda x:x # 배치 리스트 요소를 데이터 개별 인스턴스로 세팅
                            )       
    
    print("done!")
    print()
    return dataLoader # dataloader 객체 자체를 pickle 직렬화하면 안 됨

if __name__ == "__main__":

    # CUDA 체크
    print()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA 사용여부: {}".format(torch.cuda.is_available()))
    print()
    
    # kobert model loading
    print("loading skt kobert model for pretraining...")
    kobert = \
        BertForPreTraining.from_pretrained('skt/kobert-base-v1')
        # BertModel.from_pretrained('skt/kobert-base-v1')
    kobert.to(device)
    configuration = kobert.config
    print(configuration)
    print()

    # tokenizer loading
    print("skt kobert tokenizer loading...")
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    print(dir(tokenizer))
    print("vocab size: ", tokenizer.vocab_size)
    print('나 어떻게. 또 만나는 그대 모습.')
    print(tokenizer.tokenize('나 어떻게. 또 만나는 그대 모습.'))
    print(tokenizer.encode('나 어떻게. 또 만나는 그대 모습.'))
    print(tokenizer('나 어떻게. 또 만나는 그대 모습.'))
    print('...')
    print(tokenizer('나 어떻게. 또 만나는 그대 모습', '그대를 다시 보면 그대로 돌아서리.', 
                    return_tensors = 'pt', 
                    max_length = 512, 
                    truncation = True, 
                    padding = 'max_length'))
    print() 
    print(type(tokenizer.get_vocab_file())) # str of filename
    print(tokenizer.get_vocab_file())
    print(type(tokenizer.get_vocab())) # dict of vocab
    print(tokenizer.get_vocab())
    print(tokenizer.save_vocabulary('dataset/'))
    print(tokenizer.save_pretrained('dataset/'))
    print()
    # input('... press enter ...')
         
    ##########################################
    # dataframe loading and train loader setting
    with open(dataFileName[0], 'rb') as f:
        listOfTuple = pickle.load(f)
    df1 = pd.DataFrame.from_records(listOfTuple, columns = ['case_full_no', 'numberingInItem', 'phrase'])
    print(df1.info())
    print()
    
    with open(dataFileName[1], 'rb') as f:
        listOfTuple = pickle.load(f)
    df2 = pd.DataFrame.from_records(listOfTuple, columns = ['case_full_no', 'numberingInItem', 'phrase'])
    print(df2.info())
    print()
    df = pd.concat([df1, df2])
    print(df.info())
    print()

    train_loader = dataloader_factory(
                                      df, 
                                      groupby,
                                      x_dataFieldName,
                                      seq_length,
                                      device,
                                      batch_size=batch_size,
                                      from_saved=from_saved
                                      )    
    # with open('dataset/listOfListsOfTuplesForCaseGistsPhraseTrain_loader.pickle', 'wb') as f:
    #     pickle.dump(train_loader, f)
        
    # with open('dataset/listOfListsOfTuplesForCaseGistsPhraseTrain_loader.pickle', 'rb') as f:
    #     train_loader = pickle.load(f)
    print(type(train_loader))
    print(dir(train_loader))
    print()
    
    ##########################################
    
    kobert.train()
    
    # optimizer와 scheduler 설정
    t_total = len(train_loader) * epochs
    warmup_steps = int(t_total * warmup_ratio)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in kobert.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in kobert.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    scheduler = \
        get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=t_total)
    
    # optim = AdamW(kobert.parameters(), lr = learning_rate)
    for epoch in range(epochs):
        # setup loop with TQDM and dataloader
        loop = tqdm(train_loader, leave=True)
        for batch_id, batch in enumerate(loop):
            # initialize calculated gradients (from prev step)
            optimizer.zero_grad()
            # pull all tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            next_sentence_label = batch['next_sentence_label'].to(device)
            labels = batch['labels'].to(device)
            # process
            outputs = kobert(
                            input_ids, 
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            next_sentence_label=next_sentence_label,
                            labels=labels
                            )
            # extract loss
            loss = outputs.loss
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optimizer.step()          
            # Update learning rate schedule
            scheduler.step()  
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())            
                        
            if batch_id % log_interval == 0:

                with open(f"model/bert{epoch}model{batch_id}.model", 'wb') as f:
                    pickle.dump(kobert, f)
                    
    with open(f"model/bert{epoch}model{batch_id}.model", 'wb') as f:
        pickle.dump(kobert, f) 
            
'''
##########################################################################
import random

sentence_a = []
sentence_b = []
label = []

for paragraph in text:
    sentences = [
        sentence for sentence in paragraph.split('.') if sentence != ''
    ]
    num_sentences = len(sentences)
    if num_sentences > 1:
        start = random.randint(0, num_sentences-2)
        # 50/50 whether is IsNextSentence or NotNextSentence
        if random.random() >= 0.5:
            # this is IsNextSentence
            sentence_a.append(sentences[start])
            sentence_b.append(sentences[start+1])
            label.append(0)
        else:
            index = random.randint(0, bag_size-1)
            # this is NotNextSentence
            sentence_a.append(sentences[start])
            sentence_b.append(bag[index])
            label.append(1)

for i in range(3):
    print(label[i])
    print(sentence_a[i] + '\n---')
    print(sentence_b[i] + '\n')
            
inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt',
                   max_length=512, truncation=True, padding='max_length')

inputs['next_sentence_label'] = torch.LongTensor([label]).T

inputs['labels'] = inputs.input_ids.detach().clone()

# create random array of floats with equal dimensions to input_ids tensor
rand = torch.rand(inputs.input_ids.shape)
# create mask array
mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * \
           (inputs.input_ids != 102) * (inputs.input_ids != 0)
           
selection = []

for i in range(inputs.input_ids.shape[0]):
    selection.append(
        torch.flatten(mask_arr[i].nonzero()).tolist()
    )
           
for i in range(inputs.input_ids.shape[0]):
    inputs.input_ids[i, selection[i]] = 103

class OurDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)
        
dataset = OurDataset(inputs)

loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

from tqdm import tqdm  # for our progress bar

epochs = 2

for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        next_sentence_label = batch['next_sentence_label'].to(device)
        labels = batch['labels'].to(device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        next_sentence_label=next_sentence_label,
                        labels=labels)
        # extract loss
        loss = outputs.loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
##########################################################################
'''   
'''        
    # # model object setting
    # kobertmodel = BERTforFineTuning(kobert)
    # kobertmodel.to(device)
    # # kobertmodel.train()

    # print(type(iter(train_loader)))
    
    # sample = next(iter(train_loader))
    
    # input_ids, attention_mask, token_type_ids, phrase = sample
    # print(type(input_ids))
    # print(input_ids.shape) # batch size, seqence length(512)
    # print(input_ids)
    # print(type(phrase))
    # print(phrase) # tuple
    # print()
    # out = kobertmodel(input_ids, attention_mask, token_type_ids)
    # print(type(out))
    # print(out.last_hidden_state.shape) # 2, 512, 768(embedding size)
    # print(out.pooler_output.shape) # 2, 768
    # print(out)
    # print()
  
    
# dataset에서 미리 처리해주기
class BERTLanguageModelingDataset(torch.utils.data.Dataset):
    def __init__(self, data: List, vocab: spm.SentencePieceProcessor, sep_id: str='[SEP]', cls_id: str='[CLS]',
                mask_id: str='[MASK]', pad_id: str="[PAD]", seq_len: int=512, mask_frac: float=0.15, p: float=0.5):
        """Initiate language modeling dataset.
        Arguments:
            data (list): a tensor of tokens. tokens are ids after
                numericalizing the string tokens.
                torch.tensor([token_id_1, token_id_2, token_id_3, token_id1]).long()
            vocab (sentencepiece.SentencePieceProcessor): Vocabulary object used for dataset.
            p (float): probability for NSP. defaut 0.5
        """
        super(BERTLanguageModelingDataset, self).__init__()
        self.vocab = vocab
        self.data = data
        self.seq_len = seq_len
        self.sep_id = vocab.piece_to_id(sep_id)
        self.cls_id = vocab.piece_to_id(cls_id)
        self.mask_id = vocab.piece_to_id(mask_id)
        self.pad_id = vocab.piece_to_id(pad_id)
        self.p = p
        self.mask_frac = mask_frac

    def __getitem__(self, i):
        
        seq1 = self.vocab.EncodeAsIds(self.data[i].strip())
        seq2_idx = i+1
        # decide wheter use random next sentence or not for NSP task
        if random.random() > p:
            is_next = torch.tensor(1) # 연속된 문장이 아님?
            while seq2_idx == i+1:
                seq2_idx = random.randint(0, len(data))
        else:
            is_next = torch.tensor(0) # 연속된 문장임?

        seq2 = self.vocab.EncodeAsIds(self.data[seq2_idx])

        if len(seq1) + len(seq2) >= self.seq_len - 3: # except 1 [CLS] and 2 [SEP]
            idx = self.seq_len - 3 - len(seq1)
            seq2 = seq2[:idx]

        # sentence embedding: 0 for A, 1 for B
        
        mlm_target = torch.tensor([self.cls_id] + seq1 + [self.sep_id] + seq2 + [self.sep_id] + [self.pad_id] * (self.seq_len - 3 - len(seq1) - len(seq2))).long().contiguous()
        
        sent_emb = torch.ones((mlm_target.size(0)))
        _idx = len(seq1) + 2
        sent_emb[:_idx] = 0
        
        def masking(data):
            data = torch.tensor(data).long().contiguous()
            data_len = data.size(0)
            ones_num = int(data_len * self.mask_frac)
            zeros_num = data_len - ones_num
            lm_mask = torch.cat([torch.zeros(zeros_num), torch.ones(ones_num)])
            lm_mask = lm_mask[torch.randperm(data_len)]
            data = data.masked_fill(lm_mask.bool(), self.mask_id)

            return data

        mlm_train = torch.cat([torch.tensor([self.cls_id]), masking(seq1), torch.tensor([self.sep_id]), masking(seq1), torch.tensor([self.sep_id])]).long().contiguous()
        mlm_train = torch.cat([mlm_train, torch.tensor([self.pad_id] * (512 - mlm_train.size(0)))]).long().contiguous()

        # mlm_train, mlm_target, sentence embedding, NSP target
        return mlm_train, mlm_target, sent_emb, is_next
        # return self.data[i]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for x in self.data:
            yield x

    def get_vocab(self):
        return self.vocab

    def decode(self, x):
        return self.vocab.DecodeIds(x)
        
class BertModel(nn.Module):
    def __init__(self, voc_size:int=30000, seq_len: int=512, d_model: int=768, d_ff:int=3072, pad_idx: int=1,
                num_encoder: int=12, num_heads: int=12, dropout: float=0.1):
        super(BertModel, self).__init__()
        self.pad_idx = pad_idx
        self.emb = BERTEmbedding(seq_len, voc_size, d_model, dropout)
        self.encoders = Encoders(seq_len, d_model, d_ff, num_encoder, num_heads, dropout)

    def forward(self, input: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        # param:
        #     input: a batch of sequences of words
        # dim:
        #     input:
        #         input: [B, S]
        #     output:
        #         result: [B, S, V]
        pad_mask = get_attn_pad_mask(input, input, self.pad_idx)
        emb = self.emb(input, seg) # [B, S, D_model]
        output = self.encoders(emb, pad_mask) # [B, S, D_model]

        return output # [B, S, D_model]

논문에도 나와있듯 segment embedding, token embedding, positional embedding 세 개가 합쳐져서 input이 되고, 이를 transformer 인코더에 넣는 구조이다.

BERTEmbedding은 다음과 같은 구조를 갖고 있다.

class BERTEmbedding(nn.Module):
    """
    Embeddings for BERT.
    It includes segmentation embedding, token embedding and positional embedding.
    I add dropout for every embedding layer just like the original transformer.
    """
    def __init__(self, seq_len: int=512, voc_size: int=30000, d_model: int=768, dropout: float=0.1) -> None:
        super(BERTEmbedding, self).__init__()
        self.tok_emb = nn.Embedding(num_embeddings=voc_size, embedding_dim=d_model)
        self.tok_dropout = nn.Dropout(dropout)
        self.seg_emb = nn.Embedding(2, d_model)
        self.seg_dropout = nn.Dropout(dropout)
        self.pos_emb = PositionalEncoding(d_model, seq_len, dropout)

    def forward(self, tokens: torch.Tensor, seg: torch.Tensor):
        """
        tokens: [B, S]
        seg: [B, S]. seg is binary tensor. 0 indicates that the corresponding token for its index belongs sentence A
        """
        tok_emb = self.tok_emb(tokens) # [B, S, d_model]
        seg_emb = self.seg_emb(seg) # [B, S, d_model]
        pos_emb = self.pos_emb(tokens) # [B, S, d_model]

        return self.tok_dropout(tok_emb) + self.seg_dropout(seg_emb) + pos_emb  # [B, S, d_model]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: int=0.1):
        super(PositionalEncoding, self).__init__()
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        self.emb = nn.Embedding(seq_len, d_model)
        
    def forward(self, x: torch.Tensor):
        # x: [B, S]. x is tokens
        pos = torch.arange(self.seq_len, dtype=torch.long, device=x.device) # [S]
        pos = pos.unsqueeze(0).expand(x.size()) # [1, S] -> [B, S]
        pos_emb = self.emb(pos)
        return self.dropout(pos_emb) # [B, S, D_model]
        
다음은 pre-train의 task를 담당하는 MaskedLanguageModeling과 NextSentencePrediction이다. 각기 bert 몸체를 받아 동작하도록 만들어놨다.

from typing import Optional
import torch
import torch.nn as nn

class MaskedLanguageModeling(nn.Module):
    def __init__(self, bert: nn.Module, voc_size:int=30000):
        super(MaskedLanguageModeling, self).__init__()
        self.bert = bert
        d_model = bert.emb.tok_emb.weight.size(1)
        self.linear = nn.Linear(d_model, voc_size)

    def forward(self, input: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        # param:
        #     input: a batch of sequences of words
        #     seg: Segmentation embedding for input tokens
        # dim:
        #     input:
        #         input: [B, S]
        #         seg: [B, S]
        #     output:
        #         result: [B, S, V]
        # output = self.bert(input, seg) # [B, S, D_model]
        output = self.linear(output) # [B, S, voc_size]

        return output # [B, S, voc_size]

class NextSentencePrediction(nn.Module):
    def __init__(self, bert: nn.Module):
        super(NextSentencePrediction, self).__init__()
        self.bert = bert
        d_model = bert.emb.tok_emb.weight.size(1)
        self.linear = nn.Linear(d_model, 2)

    def forward(self, input: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        # param:
        #     input: a batch of sequences of words
        #     seg: Segmentation embedding for input tokens
        # dim:
        #     input:
        #         input: [B, S]
        #         seg: [B, S]
        #     output:
        #         result: [B, S, V]
        output = self.bert(input, seg) # [B, S, D_model]
        output = self.linear(output) # [B, S, 2]

        return output[:, 0, :] # [B, 2]        

import torch.nn as nn
import torch
import torch.optim as optim

def train(mlm_head: nn.Module, nsp_head: nn.Module, dataloader: torch.utils.data.DataLoader, mlm_optimizer: optim.Optimizer, nsp_optimizer: optim.Optimizer,
          criterion: nn.Module, clip: float):
    mlm_head.train()
    nsp_head.train()

    mlm_epoch_loss = 0
    nsp_epoch_loss = 0

    cnt = 0 # count length for avg loss
    for batch, (mlm_train, mlm_target, sent_emb, is_next) in enumerate(dataloader):
        # MLM task
        mlm_optimizer.zero_grad()
        mlm_output = mlm_head(mlm_train.to(DEVICE), sent_emb.to(DEVICE))
        mlm_output = mlm_output.reshape(-1, mlm_output.shape[-1])
        mlm_loss = criterion(mlm_output, mlm_target.to(DEVICE).reshape(-1)) # CE
        mlm_loss.backward()
        torch.nn.utils.clip_grad_norm_(mlm_head.parameters(), 1)
        mlm_optimizer.step()

        # NSP task
        nsp_optimizer.zero_grad()
        nsp_output = nsp_head(mlm_train.to(DEVICE), sent_emb.to(DEVICE))
        nsp_loss = criterion(nsp_output, is_next.to(DEVICE)) # no need for reshape target
        nsp_loss.backward()
        torch.nn.utils.clip_grad_norm_(nsp_head.parameters(), 1)
        nsp_optimizer.step()

        mlm_epoch_loss += mlm_loss.item()
        nsp_epoch_loss += nsp_loss.item()
        cnt += 1

    return mlm_epoch_loss / cnt, nsp_epoch_loss / cnt, 

def evaluate(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module):
    model.eval()
    mlm_epoch_loss = 0
    nsp_epoch_loss = 0

    cnt = 0 # count length for avg loss
    with torch.no_grad():
        for batch, (mlm_train, mlm_target, sent_emb, is_next) in enumerate(dataloader):
            # MLM task
            mlm_output = mlm_head(mlm_train.to(DEVICE), sent_emb.to(DEVICE))
            mlm_output = mlm_output.reshape(-1, mlm_output.shape[-1])
            mlm_loss = criterion(mlm_output, mlm_target.to(DEVICE).reshape(-1)) # CE

            # NSP task
            nsp_optimizer.zero_grad()
            nsp_output = nsp_head(mlm_train.to(DEVICE), sent_emb.to(DEVICE))
            nsp_loss = criterion(nsp_output.to(DEVICE), is_next.to(DEVICE)) # CE

            mlm_epoch_loss += mlm_loss.item()
            nsp_epoch_loss += nsp_loss.item()
            cnt += 1

    return epoch_loss / cnt

def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
    
############################################################################################################
     
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from konlpy.tag import Okt

import pandas as pd
import pickle, re

def sentences_preproc(lines): # lines는 두 열로 이루어진 dataframe / (사건번호, 토큰 리스트) 튜플이 원소인 리스트 반환
        i = 0
        preproc = []
        okt = Okt()
        for row in lines.itertuples():
            if not row.gists: 
                print(f'no item at {i}')
                continue
            if i % 500 == 0:
                print("current(every 500) - " + str(i+1))
            i += 1

            noHangul = re.compile('[^ ㄱ-ㅣ가-힣]+') # 한글과 띄어쓰기를 제외한 모든 글자
            line = noHangul.sub(' ', row.gists) # 한글과 띄어쓰기를 제외한 모든 부분을 제거

            # 형태소 분석
            malist = okt.pos(row.gists, norm=True, stem=True)
            # 필요한 어구만 대상으로 하기
            r = []
            for word in malist:
                # 어미/조사/구두점 등은 대상에서 제외 
                if not word[1] in ["Josa", "Eomi", "Punctuation"]:
                    r.append(word[0])
            preproc.append((row.case_full_no, r))
        print(f'totally {i} lines')
        return preproc

class Doc2VecCorpus:
    def __init__(self, preproc): 
        self.sentenceList = preproc # sentenceList는 (사건명, 단어 리스트) 열의 list
    def __iter__(self):
        for idx, text in self.sentenceList:
            yield TaggedDocument(
                words = text, # words는 한 sentence 단어들의 리스트
                tags = ['%s' % idx]) # idx는 case_full_no

if __name__ == '__main__':

    s_no = 200000

    print("# 데이터프레임 준비")
    df_corpus = pd.read_pickle('C:/Users/hcjeo/VSCodeProjects/web2df/saved/df_corpus.pickle')
    print(df_corpus.info())
    df_summary = pd.read_pickle('C:/Users/hcjeo/VSCodeProjects/web2df/saved/df_summary.pickle')
    print(df_summary.info())

    print("# sen2vec preproc")
    # text_data = df_corpus[['case_full_no','reasoning']][:s_no]
    text_data = df_summary[['case_full_no','gists']][:s_no] # lines는 두 열로 이루어진 dataframe 
    preproc = sentences_preproc(text_data) # (사건번호, 토큰 리스트) 튜플이 원소인 리스트 반환
    # with open('word2vec.preproc', 'rb') as f:
    #     preproc = pickle.load(f) # 단어의 리스트의 리스트
    with open('preproc_sen2vec.preproc', 'wb') as f:
        pickle.dump(preproc, f)

    print("# sen2vec embedding")
    sen2vec_corpus = Doc2VecCorpus(preproc) # preproc - 사건명과 단어의 리스트 튜블의 리스트
    sen2vec_model = Doc2Vec(sen2vec_corpus)

    print("# sen2vec storing")
    with open('sen2vec.model', 'wb') as f:
        pickle.dump(sen2vec_model, f)
    with open('sen2vec.model', 'rb') as f:
        sen2vec_model = pickle.load(f)

    print("# sen2vec example")
    sen2vec_model.docvecs.most_similar('이행지체', topn=5)

    # for idx, doctag in sorted(doc2vec_model.docvecs.doctags.items(), key=lambda x:x[1].offset):
    #     print(idx, doctag)
    # https://lovit.github.io/nlp/representation/2018/03/26/word_doc_embedding/
'''

# from gensim.models import word2vec
# from konlpy.tag import Okt

# import re
# import numpy as np
# # 판결요지 임베딩 for 판례 추천

# import pickle
# import pandas as pd

# if __name__ == '__main__':
#     print("# word2vec retrieving")
#     with open('word2vec.model', 'rb') as f:
#         model = pickle.load(f)
#     with open('word2vec.preproc', 'rb') as f:
#         preproc = pickle.load(f)

#     # 문장 임베딩: 단어벡터 평균
#     word_vectors = model.wv
#     sentence_vector_list = []

#     for line in preproc:
#         word_vectors_list_for_an_item = np.empty((100, 0), float)
#         for v in line:
#             if v in word_vectors.key_to_index:
#                 word_vectors_list_for_an_item = np.append(word_vectors_list_for_an_item, word_vectors[v].reshape(100, 1), axis = 1)
#             else:
#                 word_vectors_list_for_an_item = np.append(word_vectors_list_for_an_item, word_vectors['판결'].reshape(100, 1), axis = 1)
#         word_vectors_list_for_an_item_mean = word_vectors_list_for_an_item.mean(axis=1)
#         sentence_vector_list.append(list(word_vectors_list_for_an_item_mean))

#     print(type(sentence_vector_list))
#     sentence_vector_ndarray = np.array(sentence_vector_list)
#     print(np.array(sentence_vector_ndarray).shape)
#     print(sentence_vector_ndarray[~np.isnan(sentence_vector_ndarray)].shape)

#     sentence_vector_list = list(sentence_vector_ndarray[~np.isnan(sentence_vector_ndarray)].reshape(115687, 100))
#     print(np.array(sentence_vector_list).shape)
#     print(sentence_vector_list[0:2])
#     sentences = []
#     for line in sentence_vector_list:
#         sentences.append(list(line))
#     sentence_vector_list = sentences
#     print(sentence_vector_list[0:2])