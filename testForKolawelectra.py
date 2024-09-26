# from tqdm import tqdm
# from collections import defaultdict
# from sklearn.model_selection import train_test_split

from transformers import ElectraForMaskedLM # generator class
from transformers import ElectraForPreTraining # discriminator class
from transformers import ElectraTokenizer #  gen disc 공통
from transformers import ElectraModel
# from transformers.optimization import get_cosine_schedule_with_warmup

# from torch.utils.data import Dataset, DataLoader
# from torch.nn import CrossEntropyLoss
# from torch.optim import AdamW
# from torch import nn

import torch
# import torch.nn.functional as F
# import gc

# import pickle
# import numpy as np
# import pandas as pd

import __case_CorpusRsSummaryGs_kolawBERT_PT_torch_monologg_koelectrabasev3 as mykolawelec

if __name__ == '__main__':
    
    isTest = False
    print('\ntokenizer for discriminator downloading: \n')
    tokenizer =\
        ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")    
    
    if isTest == True:

        PATH = r'.\\model\\koelectra_model_epch_1_btch_131899_2023-02-24-18-57-03.pth'
        seq_length = 512

        print('\ngenerator downloading: \n')
        generatorm =\
            ElectraForMaskedLM.from_pretrained("monologg/koelectra-base-v3-generator")
        print('\ngenerator config: \n', generatorm.config)

        print('\ndiscriminator monologg downloading: \n')
        discriminatorm =\
            ElectraForPreTraining.from_pretrained("monologg/koelectra-base-v3-discriminator")
        print('\ndiscriminator config: \n', discriminatorm.config)

        print('\ndiscriminator kolawelectra loading: \n')
        kolawelectra = mykolawelec.ELECTRAModel(generatorm, discriminatorm, tokenizer)
        ########################################
        kolawelectra.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
        ########################################

        # discriminator = ko electra discriminator model with binary classification top
        discriminator = kolawelectra.discriminator
        print('\ndiscriminator law config: \n', discriminator.config)
        discriminator.eval()
        
        print('\ngenerator kolawelectra loading: \n') 
        generator = kolawelectra.generator
        generator.eval()

        print()
        print("ko law electra testing....")
        print()

        print(tokenizer.SPECIAL_TOKENS_ATTRIBUTES)
        print("\n bos token: \n")
        print(tokenizer.bos_token_id)
        print("\n eos token: \n")
        print(tokenizer.eos_token_id)
        print("\n unk token: \n")
        print(tokenizer.unk_token_id)
        print("\n sep token: \n")
        print(tokenizer.sep_token_id)
        print("\n pad token: \n")
        print(tokenizer.pad_token_id)
        print("\n cls token: \n")
        print(tokenizer.cls_token_id)
        print("\n mask token: \n")
        print(tokenizer.mask_token_id)
        print("\n additional special tokens: \n")
        print(tokenizer.additional_special_tokens_ids)
        print()
        
        senSample = '''
        위 경매절차에서 집행관이 이 사건 주택에 관한 현황을 조사할 당시 피고의 처인 소외 2는 위 임차기간이 '1995. 7. 31.부터 1996년 8월 현재까지'라고 진술하여 그에 따라 집행관 명의의 1996. 8. 24.자 부동산현황조사보고서가 작성되었으며, 한편 피고는 1996. 8. 31. 위 경매법원에 이 사건 주택에 대한 확정일자부 임차인으로서 권리신고 및 임차보증금 40,000,000원에 대한 배당요구를 함에 있어서, 임대차기간을 1996. 7. 30.까지로 하여 작성된 1995. 7. 29.자 임대차계약서만을 제출하였을 뿐 임대차기간의 연장에 관한 아무런 자료를 제출하지 않았고, 경매법원은 피고의 위와 같은 배당요구사실을 소유자인 소외 1에게 통지하지는 않았고 다만 원고가 낙찰받기 직전의 입찰명령에 첨부된 부동산목록에 피고가 임차인으로 된 [MASK]의 기간을 1996. 7. 30.까지로 표시하였다.''' # 임대차 masked
        senSampleLabel = '''
        위 경매절차에서 집행관이 이 사건 주택에 관한 현황을 조사할 당시 피고의 처인 소외 2는 위 임차기간이 '1995. 7. 31.부터 1996년 8월 현재까지'라고 진술하여 그에 따라 집행관 명의의 1996. 8. 24.자 부동산현황조사보고서가 작성되었으며, 한편 피고는 1996. 8. 31. 위 경매법원에 이 사건 주택에 대한 확정일자부 임차인으로서 권리신고 및 임차보증금 40,000,000원에 대한 배당요구를 함에 있어서, 임대차기간을 1996. 7. 30.까지로 하여 작성된 1995. 7. 29.자 임대차계약서만을 제출하였을 뿐 임대차기간의 연장에 관한 아무런 자료를 제출하지 않았고, 경매법원은 피고의 위와 같은 배당요구사실을 소유자인 소외 1에게 통지하지는 않았고 다만 원고가 낙찰받기 직전의 입찰명령에 첨부된 부동산목록에 피고가 임차인으로 된 임대차의 기간을 1996. 7. 30.까지로 표시하였다.'''

        print('\ntokenized result with "tokenizer.tokenize" with no option for :' + senSample + "\n")
        print(tokenizer.tokenize(senSample))
        print('\ninput_ids for ' + senSample + " with tensored result of 'tokenizer.encode' as well as 'add_special_tokens=True': \n")
        input_ids = torch.tensor(tokenizer.encode(senSample, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        print(input_ids)
        print('input_ids type and shape: ')
        print(type(input_ids))
        print(input_ids.shape)
        # input('...')

        inputs = tokenizer(senSample, return_tensors="pt")
        print('\ntokenized masked with "tokenizer for pt option": \n')
        print(inputs["input_ids"])
        print(type(inputs))
        print(torch.tensor(inputs['input_ids'], dtype=torch.int32).shape)
        # input('...')

        labels = tokenizer(senSampleLabel, return_tensors="pt")["input_ids"]
        print('\ntokenized label with "tokenizer for pt option": \n')
        print(labels)
        print(type(labels))
        print(torch.tensor(labels, dtype=torch.int32).shape)
        # input('...')
        
        ####################
        logits = generator(input_ids).logits
        print('\ngenerator logits for sample sentence masked: \n', logits)
        print('\ntype and shape of logits: \n')
        print(type(logits))
        print(logits.shape)
        ####################
        print('\ngenerator outputs with "**inputs agrs" for sample sentence masked: \n')
        outputs = generator(**inputs)
        print(type(outputs))
        print(dir(outputs))
        print('\nlogits: \n')
        print(outputs.logits)
        print(outputs.logits.shape)
        print('\nloss: \n')
        print(outputs.loss)
        ####################
        # input('...*...')

        logits = discriminator(labels).logits # labels = tokenizer(senSampleLabel, return_tensors="pt")["input_ids"]
        print('\ndiscriminator logits for sample sentence masked(nonsensical): \n', logits)
        print('\ntype and shape of logits: \n')
        print(type(logits))
        print(logits.shape)
        # input('...')

        print('\ndiscriminator loss for sample sentence masked against labels(nonsensical): \n')
        outputs = discriminator(**inputs, labels=labels) # **inputs for masked senSample
        print('\noutputs: \n')
        print(dir(outputs))
        loss = outputs.loss
        print('\nloss: ', loss)
        print(type(loss))
        print(loss.shape)
        # input('...')

        sentence = ["나는 내일 밥을 먹었다.", "너는 내일 밥을 먹어라."]
        print('\ntesting for:')
        print(sentence)
        tokens = tokenizer.tokenize(sentence[0])
        print('\ntokenized with tokenizer.tokenize for first sentence: \n')
        print(tokens)
        inputs = tokenizer.encode(sentence[0], return_tensors="pt")
        print('\ntokenized with tokenizer.encode with pt option for first sentence: \n')
        print(inputs)

        inputss = tokenizer(
            sentence,
            return_tensors='pt',
            max_length=seq_length, 
            truncation=True, 
            padding='max_length'
            )
        
        print('\ntokenized with tokenizer with pt option for 2 sens: \n')
        print(inputss)
        print(type(inputss))
        print(inputss.input_ids.dtype)
        print(dir(inputss))

        # print('\ndiscriminator output for the above 2 sens inputss: \n')
        # print(dir(discriminator(inputss))) <================================ERROR!!!!!!!!!! 입력을 list로 할 수 없는 것으로 보임
        # print(discriminator(inputss).hidden_states)  # None
        # print(discriminator(inputss).items) # built-in method items of ElectraForPreTrainingOutput object
        # print(discriminator(inputss).keys) # built-in method keys of ElectraForPreTrainingOutput object
        # print(discriminator(inputss).loss) # None
        # print(discriminator(inputss).values) # built-in method of ElectraForPreTrainingOutput object
        # ####################
        # print(discriminator(inputss).logits) # rank 2 torch tensor object
        # ####################

        print('\ntokenized with tokenizer only by two args of 2 sens: \n')
        inputs = tokenizer(sentence[0], sentence[1])
        print(inputs) # {'input_ids': [2, 2236, 4034, 8258, 2739, 4292, 2654, 4480, 4176, 18, 3, 2267, 4034, 8258, 2739, 4292, 2654, 4025, 4118, 18, 3], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
        # input('...**...')

        print()
        print('ko law electra discriminator testing....\n')

        fake_sentence = "원고가 낙찰받기 직전의 입찰명령에 첨부된 부동산목록에 피고가 임차인으로 된 임대차의 기간을 1996. 7. 30.까지로 붕괴하였다."
        print('fake sentence: ')
        print(fake_sentence)
        fake_tokens = tokenizer.tokenize(fake_sentence)
        print(fake_tokens)
        # ['그', '빠른', '갈색', '여우', '##가', '그', '게으', '##른', '개', '아래', '##를', '뛰어넘', '##었', '##다', '.']
        fake_inputs = tokenizer.encode(fake_sentence, return_tensors="pt")
        print('fake inputs: ')
        print(fake_inputs)
        print()

        ####################
        discriminator_outputs = discriminator(fake_inputs)
        print('discriminator outputs: ')
        print(dir(discriminator_outputs))
        print(discriminator_outputs.logits) # print(discriminator_outputs[0]) 완전히 동일
        print(discriminator_outputs[0])
        # input("...***...")

        ####################
        predictions = torch.round((torch.sign(discriminator_outputs[0]) + 1) / 2)
        print(predictions.squeeze().tolist())
        # [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ####################

        tpl = discriminator(fake_inputs, output_hidden_states=True)
        for i, x in enumerate(tpl):
            print(f'\n{i} : {x}\n')
        
        print(type(tpl[1])) # tpl[1]도 tuple
        print(type(tpl.hidden_states)) # tpl[1]은 tpl.hidden_states와 완전히 동일
        print(dir(tpl[1])) 
        print(tpl[1])
        print(tpl[1][-1].shape) # 1, 17, 768 최종 representation matrix

        discriminator.save_pretrained('.//model//koelectra_model_epch_1_btch_131899_2023-02-24-18-57-03//')

    model = ElectraModel.from_pretrained('.//model//koelectra_model_epch_1_btch_131899_2023-02-24-18-57-03//')
    model.eval()

    sentence = "원고가 낙찰받기 직전의 입찰명령에 첨부된 부동산목록에 피고가 임차인으로 된 임대차의 기간을 1996. 7. 30.까지로 표시하였다."
    print('\nreal sentence: ')
    print(sentence)
    real_tokens = tokenizer.tokenize(sentence)
    print(real_tokens)
    real_inputs = tokenizer.encode(sentence, return_tensors="pt")
    print('\nreal inputs: ')
    print(real_inputs)
    print()


    output = model(real_inputs)
    print(output)
    print(dir(output))
    print(output.last_hidden_state.shape)

    