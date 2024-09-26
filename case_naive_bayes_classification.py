import math, sys
import pandas as pd
from konlpy.tag import Okt

words = set()
doc_word_list = []
word_dict = {}
# topic = {}
topic_dict = {}

# 그 다음 문서의 전처리를 구현합니다.
def preproc(doc):
    doc_word_list = []
    okt = Okt()
    word_pos_dic_list = okt.pos(doc, norm=True, stem=True)
    for word in word_pos_dic_list: 
        if not word[1] in ["Josa", "Eomi", "Punctuation"]:
            doc_word_list.append(word[0])
    return doc_word_list

# 위에서 본 첫째 표를 구현합니다.
def topic_frequency(topic, topic_dict):
        
    if not topic in topic_dict:
        topic_dict[topic] = 0
    topic_dict[topic] += 1

    return topic_dict

# 위에서 본 두 번째 표를 파이썬 딕셔너리를 이용해 구현합니다.
def topic_word_table(word, topic, word_dict, words):

    if not topic in word_dict:
        word_dict[topic] = {}
    if not word in word_dict[topic]:
        word_dict[topic][word] = 0
    word_dict[topic][word] += 1
    words.add(word)

    return words, word_dict

# 문서 하나와 그 해당 주제를 입력하면 위 두 함수들을 이용해서 누적적으로 문서들, 즉 말뭉치의 주제별 문서 분포와 주제별 단어 분포, 즉 위 두 표를 업데이트하는 함수를 정의합니다.
def corpus_update(doc, topic, topic_dict, word_dict, words):
        
    word_list = preproc(doc)
    for word in word_list:
        words, word_dict = topic_word_table(word, topic, word_dict, words)
    topic_dict = topic_frequency(topic, topic_dict)

    return words, word_dict, topic_dict

# 이제 사전확률분포 P( θ )를 파이썬 함수로 구현합니다.
def prior(topic, topic_dict):

    sum_topics = sum(topic_dict.values())
    topic_freq = topic_dict[topic]
    return topic_freq / sum_topics

# 그리고 주제 조건부 단어출현 확률분포함수 부분을 파이썬 함수로 구현합니다.
def likelihood(word, words, word_dict, topic, word_count):
    n = word_count(word, topic, word_dict) + 1 # 라플라스 정규화
    d = sum(word_dict[topic].values()) + len(words)
    return n / d

def word_count(word, topic, word_dict):
    if word in word_dict[topic]:
        return word_dict[topic][word]
    else:
        return 0

# 또 사후확률분포 역시 파이썬 함수로 구현합니다.
def posterior(words, topic, topic_dict, prior, word_dict, word_count):
    score = math.log(prior(topic, topic_dict))
    for word in words:
        score += math.log(likelihood(word, words, word_dict, topic, word_count))
    return score

# 마지막으로 새로운 문서를 입력하여 주제별 확률분포를 구하는 함수를 구현합니다.
def predict(doc, topic_dict, preproc, posterior, prior, word_count):
    best_topic = None
    max_score = -sys.maxsize 
    words = preproc(doc) 
    score_list = []
    for topic in topic_dict.keys():
        score = posterior(words, topic, topic_dict, prior, word_dict, word_count)
        score_list.append((topic, score))
        if score > max_score:
            max_score = score
            best_topic = topic
    return best_topic, score_list

if __name__=='__main__':
    
    # df = pd.read_csv('./dataset/dev.tsv', sep='\t')
    df = pd.read_csv('C:/Users\hcjeo\VSCodeProjects/_python_practice/dataset/dfFinal.csv')
    df_train = df[:10000]
    df_test = df[10001:]
    for i in range(len(df_train)):
        if i == 0: continue
        print(i)
        words, word_dict, topic_dict = corpus_update(df.iloc[i, 2], df.iloc[i, 1], topic_dict, word_dict, words)

    for i in range(len(df_test)):
        if i == 0: continue
        print(df.iloc[i, :])
        best_topic, score_list = predict(df.iloc[i, 2], topic_dict, preproc, posterior, prior, word_count)
        print(best_topic)
        print(score_list)
        print('right answer: ', df.iloc[i, 1])
        answer = input('press \'z\' and enter for quit: ')
        if answer == 'z':
            break