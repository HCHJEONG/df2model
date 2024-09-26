# from tqdm import tqdm
# from collections import defaultdict
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score, accuracy_score

# from transformers import ElectraForMaskedLM # generator class
# from transformers import ElectraForPreTraining # discriminator class
# from transformers import ElectraTokenizer #  gen disc 공통
# from transformers.optimization import get_cosine_schedule_with_warmup

# from torch.utils.data import Dataset, DataLoader
# from torch.nn import CrossEntropyLoss
# from torch.optim import AdamW
# from torch import nn

# import torch
# import torch.nn.functional as F
# import gc

# import pickle
# import datetime
# import numpy as np
# import pandas as pd

# python 3 8 11 환경에서 모두 다시 설치해야 함
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
nltk.download('punkt')

model_dir = "lcw99/t5-large-korean-text-summary"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

max_input_length = 512 + 256

text = """
주인공 강인구(하정우)는 ‘수리남에서 홍어가 많이 나는데 다 갖다버린다’는 친구 
박응수(현봉식)의 얘기를 듣고 수리남산 홍어를 한국에 수출하기 위해 수리남으로 간다. 
국립수산과학원 측은 “실제로 남대서양에 홍어가 많이 살고 아르헨티나를 비롯한 남미 국가에서 홍어가 많이 잡힌다”며 
“수리남 연안에도 홍어가 많이 서식할 것”이라고 설명했다.

그러나 관세청에 따르면 한국에 수리남산 홍어가 수입된 적은 없다. 
일각에선 “돈을 벌기 위해 수리남산 홍어를 구하러 간 설정은 개연성이 떨어진다”는 지적도 한다. 
드라마 배경이 된 2008~2010년에는 이미 국내에 아르헨티나, 칠레, 미국 등 아메리카산 홍어가 수입되고 있었기 때문이다. 
실제 조봉행 체포 작전에 협조했던 ‘협력자 K씨’도 홍어 사업이 아니라 수리남에 선박용 특수용접봉을 파는 사업을 하러 수리남에 갔었다.
"""

inputs = ["summarize: " + text]

inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=100)
decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]
# print(predicted_title)
print(decoded_output)
# https://cryptosalamander.tistory.com/140 nltk punkt 토크나이저 customization 글