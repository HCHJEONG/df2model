import pandas as pd
import numpy as np
import pickle

from konlpy.tag import Okt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

# 데이터프레임 준비
df_corpus = pd.read_pickle('C:/Users/hcjeo/VSCodeProjects/web2df/saved/df_corpus.pickle')
print(df_corpus.info())
df_summary = pd.read_pickle('C:/Users/hcjeo/VSCodeProjects/web2df/saved/df_summary.pickle')
print(df_summary.info())

print("# word2vec retrieving")
with open('word2vec_256.model', 'rb') as f:
    model = pickle.load(f)
with open('preproc_word2vec.preproc', 'rb') as f:
    preproc = pickle.load(f)

# 문장 임베딩: 단어벡터 평균
word_vectors = model.wv
sentence_vector_list = []

for line in preproc:
    word_vectors_list_for_an_item = np.empty((100, 0), float)
    for v in line:
        if v in word_vectors.key_to_index:
            word_vectors_list_for_an_item = np.append(word_vectors_list_for_an_item, word_vectors[v].reshape(100, 1), axis = 1)
        else:
            word_vectors_list_for_an_item = np.append(word_vectors_list_for_an_item, word_vectors['판결'].reshape(100, 1), axis = 1)
    word_vectors_list_for_an_item_mean = word_vectors_list_for_an_item.mean(axis=1)
    sentence_vector_list.append(list(word_vectors_list_for_an_item_mean))

print(type(sentence_vector_list))
sentence_vector_ndarray = np.array(sentence_vector_list)
print(np.array(sentence_vector_ndarray).shape)
print(sentence_vector_ndarray[~np.isnan(sentence_vector_ndarray)].shape)

sentence_vector_list = list(sentence_vector_ndarray[~np.isnan(sentence_vector_ndarray)].reshape(115687, 100))
print(np.array(sentence_vector_list).shape)
print(sentence_vector_list[0:2])
sentences = []
for line in sentence_vector_list:
    sentences.append(list(line))
sentence_vector_list = sentences
print(sentence_vector_list[0:2])

# KMEANS 군집 wv mean for K neighbor / 비지도인 KMEANS로 라벨링을 한 후 실제로 이 라벨링이 지도학습 모델인 KNN으로 학습이 되는지를 테스트해봄 
sentence_vector_list = np.array(sentence_vector_list)

for k in range(3, 7):

    for no in range(1, 4):
        
        print(sentence_vector_list.shape)
        kmeans = KMeans(n_clusters=k, random_state=42)
        y_pred = kmeans.fit_predict(sentence_vector_list)
        labels = list(y_pred)
        # item_ =  list(df_summary['items'][0:len(labels)])
        # series = pd.Series(item_, index = labels)
        # df = series.to_frame().reset_index()
        # print(df.head(10))
        # df.to_csv("df_500_kmeans_wv.csv", encoding='utf-8-sig')

        t, te, tl, tel = train_test_split(np.array(sentence_vector_list, dtype = object), np.array(labels, dtype= np.int32), test_size = 0.25, random_state=42)
        knn = KNeighborsClassifier(n_neighbors = no)
        knn.fit(t, tl)
        predict_label = knn.predict(te)
        print('k: ', k)
        print('n: ', no)
        print('test accuracy {:.2f}'.format(np.mean(predict_label == tel)))

# 문장 임베딩: TF IDF
# tfidf_vectorizer = TfidfVectorizer()
# tfidf_vectorizer.fit(text_data)
# sentence = [text_data[3]]
# print(sentence)
# for t in tfidf_vectorizer.transform(sentence).toarray():
#     for tt in t:
#         print(tt)
#     print("++++++++++++++++++")

# text_data_ = wvec.sentences_preproc(text_data)
# for i in range(len(text_data_)):
#     text_data_[i] = [' '.join(text_data_[i])]

# print(text_data_[3])

# sentence_vector_list_ = np.array([tfidf_vectorizer.transform([x]).toarray() for x in text_data])
# print(sentence_vector_list_.shape)
# sentence_vector_list_ = sentence_vector_list_.reshape(sentence_vector_list_.shape[0], sentence_vector_list_.shape[2])
# print(np.array(sentence_vector_list_).shape)

# KMEANS 군집 tf idf for K neighbor
# k = 5
# kmeans = KMeans(n_clusters=k, random_state=42)
# y_pred = kmeans.fit_predict(sentence_vector_list_)

# labels = list(y_pred)
# # item_ =  list(df_summary['items'][0:s_no])
# # series = pd.Series(item_, index = labels)
# # df = series.to_frame().reset_index()
# # print(df.head(10))
# # df.to_csv("df_500_kmeans_tfidf.csv", encoding='utf-8-sig')

# t, te, tl, tel = train_test_split(np.array(sentence_vector_list_, dtype = object), np.array(labels, dtype= np.int32), test_size = 0.25, random_state=42)
# knn = KNeighborsClassifier(n_neighbors = 1)
# knn.fit(t, tl)
# predict_label = knn.predict(te)
# print('test accuracy {:.2f}'.format(np.mean(predict_label == tel)))