import pickle, torch
from transformers import BartModel
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration

kobart_tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-base-v2")
print("안녕하세요. 한국어 BART 입니다.🤣:)l^o")
print(kobart_tokenizer.tokenize("안녕하세요. 한국어 BART 입니다.🤣:)l^o"))
print(kobart_tokenizer.encode("안녕하세요. 한국어 BART 입니다.🤣:)l^o"))
print()
with open('kobartbasev2tokenizer.pickle', 'wb') as f:
        pickle.dump(kobart_tokenizer, f, pickle.HIGHEST_PROTOCOL)
with open('kobartbasev2tokenizer.pickle', 'rb') as f:
    kobart_tokenizer = pickle.load(f)

model = BartModel.from_pretrained("gogamza/kobart-base-v2")
inputs = kobart_tokenizer(['안녕하세요.'], return_tensors='pt')
output_ids = model(inputs['input_ids'])
print("안녕하세요.")
print(output_ids)
print()
with open('kobartbasev2model.pickle', 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
with open('kobartbasev2model.pickle', 'rb') as f:
    model = pickle.load(f)

# pytorch lightning based model 
modelForFineTuning = BartForConditionalGeneration.from_pretrained("gogamza/kobart-base-v2") # gogamza/kobart-summarization
def summarize(text):
    input_ids =  [kobart_tokenizer.bos_token_id] + kobart_tokenizer.encode(text) + [kobart_tokenizer.eos_token_id]
    res_ids = modelForFineTuning.generate(torch.tensor([input_ids]),
                                        max_length=100,
                                        num_beams=5,
                                        eos_token_id=kobart_tokenizer.eos_token_id,
                                        bad_words_ids=[[kobart_tokenizer.unk_token_id]]
                                        )        
    a = kobart_tokenizer.batch_decode(res_ids)
    return a

print("안녕하세요.")
print(summarize("안녕하세요."))

'''
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

tok = BartTokenizer.from_pretrained("facebook/bart-large")
model = BartForConditionalGeneration(BartConfig())

input_string = "My dog is <mask> </s>"
decoder_input_string = "<s> My dog is cute"
labels_string = "My dog is cute </s>"

input_ids = tok(input_string, add_special_tokens=False, return_tensors="pt").input_ids
decoder_input_ids =tok(decoder_input_string, add_special_tokens=False, return_tensors="pt").input_ids
labels = tok(labels_string, add_special_tokens=False, return_tensors="pt").input_ids
 
loss = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels)[0]
'''