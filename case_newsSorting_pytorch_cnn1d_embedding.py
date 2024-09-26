import os
from argparse import Namespace
from collections import Counter
import json
import re
import string
import collections
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tqdm
import pickle
import gensim
from konlpy.tag import Okt

args = Namespace(
    # 날짜와 경로 정보
    news_csv_="./dfFinal_.csv", # 전처리 되고 분할 안 된 자료
    news_csv="./dfFinal.csv", # 전처리와 훈련/검증/테스트 분할이 완료된 자료
    proportion_subset_of_train=1.0,
    train_proportion=0.7,
    val_proportion=0.15,
    test_proportion=0.15,
    vectorizer_file="vectorizer.json",
    model_state_file="pytorchembedding_model.pth",
    save_dir="./",
    # 모델 하이퍼파라미터
    w2v_filepath='./word2vec.model', 
    embedding_size=100, 
    hidden_dim=100, 
    num_channels=100, 
    # 훈련 하이퍼파라미터
    seed=1337, 
    learning_rate=0.001, 
    dropout_p=0.1, 
    batch_size=8, 
    num_epochs=0,
    early_stopping_criteria=5, 
    # 실행 옵션
    cuda=False, 
    expand_filepaths_to_save_dir=True,
    catch_keyboard_interrupt=True, 
    use_w2v=False, #### Test/Learning Selection
    reload_from_files=True, ##### Test/Learning Selection
    finalTest = True #### Test/Learning Selection
    # dfFinal.csv / vectorizer.json / model.pth / word2vec.model
) 

def ko_sentences_preproc(lines: list) -> list : # from konlpy.tag import Okt
        print(f'totally {len(lines)} lines')
        i = 0
        sentences = []
        okt = Okt()
        for line in lines:
            if not line: 
                print(f'no item at {i}')
                continue
            if i % 500 == 0:
                print("current(every 500) - " + str(i+1))
            i += 1

            nonhangul = re.compile('[^ ㄱ-ㅣ가-힣]+') # 한글과 띄어쓰기를 제외한 모든 글자
            line = nonhangul.sub(' ', line) # 한글과 띄어쓰기를 제외한 모든 부분을 제거

            # 형태소 분석
            malist = okt.pos(line, norm=True, stem=True)
            # 필요한 어구만 대상으로 하기
            r = []
            for word in malist:
                # 어미/조사/구두점 등은 대상에서 제외 
                if not word[1] in ["Josa", "Eomi", "Punctuation"]:
                    r.append(word[0])
            sentences.append(" ".join(r))
        print(f'totally {i} lines')
        return sentences

def final_reviews_maker(args):
    # 원본 데이터를 읽습니다
    train_reviews = pd.read_csv(args.news_csv_) # 한글 전처리된 csv
    train_reviews = train_reviews.dropna()

    print("\n train reviews based on dfFinal csv: \n")
    print(train_reviews.info())
  
    # 클래스 비율이 바뀌지 않도록 서브셋을 만듭니다
    by_rating = collections.defaultdict(list)
    for _, row in train_reviews.iterrows():
        by_rating[row.case_sort].append(row.to_dict())
    review_subset = []
    for _, item_list in sorted(by_rating.items()):
        n_total = len(item_list)
        n_subset = int(args.proportion_subset_of_train * n_total)
        review_subset.extend(item_list[:n_subset])
    review_subset = pd.DataFrame(review_subset)
    print("\n review subset based on train reviews: \n")
    print(train_reviews.info())
    print(review_subset.head())

    # 훈련, 검증, 테스트를 만들기 위해 클래스를 기준으로 나눕니다
    by_rating = collections.defaultdict(list)
    for _, row in review_subset.iterrows():
        by_rating[row.case_sort].append(row.to_dict())
    final_list = []
    np.random.seed(args.seed)
    for _, item_list in sorted(by_rating.items()):
        np.random.shuffle(item_list)
        n_total = len(item_list)
        n_train = int(args.train_proportion * n_total)
        n_val = int(args.val_proportion * n_total)
        n_test = int(args.test_proportion * n_total)
        for item in item_list[:n_train]:
            item['split'] = 'train'
        for item in item_list[n_train:n_train+n_val]:
            item['split'] = 'val'
        for item in item_list[n_train+n_val:n_train+n_val+n_test]:
            item['split'] = 'test'
        final_list.extend(item_list)

    # 분할 데이터를 데이터 프레임으로 만듭니다
    final_reviews = pd.DataFrame(final_list)
    print("\n final review based on review subset: \n")
    print(final_reviews.info())
    print(final_reviews.head())
    print()
   
    final_reviews.to_csv(args.news_csv, index=False) # 전처리와 분할된 데이터프레임
    return final_reviews
            
def load_w2v_from_file(w2v_filepath):
# def load_w2v_from_file(w2v_filepath, words):
    """w2v 임베딩 로드 
    
    매개변수:
        w2v_filepath (str): 임베딩 파일 경로 
    반환값:
        word_to_index (dict), embeddings (numpy.ndarary)
    """

    # word_to_index = {}
    # embeddings = []
    with open(w2v_filepath, "rb") as fp:
        wvmodel = pickle.load(fp)
        # for index, word in enumerate(words):
        #     word_to_index[word] = index
        #     if type(wvmodel.wv[word]) == 'numpy.ndarray':
        #         embedding_i =  wvmodel.wv[word]
        #     else:
        #         embedding_i = np.zeros(100)
        #     embeddings.append(embedding_i)
    return wvmodel
    # return word_to_index, np.stack(embeddings)

def make_embedding_matrix(w2v_filepath, vectorizer):
    """
    특정 단어 집합에 대한 임베딩 행렬을 만듭니다.
    
    매개변수:
        w2v_filepath (str): 임베딩 파일 경로
        words (list): 단어 리스트
    """
    words = vectorizer.title_vocab._token_to_idx.keys()
    wvmodel = load_w2v_from_file(w2v_filepath)
    embedding_size = len(list(wvmodel.wv.index_to_key))
    print(f"wv embedding size: {embedding_size}")

    final_embeddings = np.zeros((len(words), args.embedding_size))

    for i, word in enumerate(words):
        if word in list(wvmodel.wv.index_to_key):
            # print(np.array(wvmodel.wv[word]).shape)
            final_embeddings[i, :] = np.array(wvmodel.wv[word])
        else:
            embedding_i = torch.ones(1, args.embedding_size)
            torch.nn.init.xavier_uniform_(embedding_i)
            final_embeddings[i, :] = embedding_i.numpy()
            # print(embedding_i.numpy().shape)
            # print(type(embedding_i.numpy()))

    return final_embeddings
 
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
        
    def to_serializable(self):
        """ 직렬화할 수 있는 딕셔너리를 반환합니다 """
        return {'token_to_idx': self._token_to_idx}

    @classmethod
    def from_serializable(cls, contents):
        """ 직렬화된 딕셔너리에서 Vocabulary 객체를 만듭니다 """
        return cls(**contents)

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

class SequenceVocabulary(Vocabulary):
    def __init__(self, token_to_idx=None, unk_token="<UNK>",
                 mask_token="<MASK>", begin_seq_token="<BEGIN>",
                 end_seq_token="<END>"):

        super(SequenceVocabulary, self).__init__(token_to_idx)

        self._mask_token = mask_token
        self._unk_token = unk_token
        self._begin_seq_token = begin_seq_token
        self._end_seq_token = end_seq_token

        self.mask_index = self.add_token(self._mask_token)
        self.unk_index = self.add_token(self._unk_token)
        self.begin_seq_index = self.add_token(self._begin_seq_token)
        self.end_seq_index = self.add_token(self._end_seq_token)

    def to_serializable(self):
        contents = super(SequenceVocabulary, self).to_serializable()
        contents.update({'unk_token': self._unk_token,
                         'mask_token': self._mask_token,
                         'begin_seq_token': self._begin_seq_token,
                         'end_seq_token': self._end_seq_token})
        return contents

    def lookup_token(self, token):
        """ 토큰에 대응하는 인덱스를 추출합니다.
        토큰이 없으면 UNK 인덱스를 반환합니다.
        
        매개변수:
            token (str): 찾을 토큰 
        반환값:
            index (int): 토큰에 해당하는 인덱스
        노트:
            UNK 토큰을 사용하려면 (Vocabulary에 추가하기 위해)
            `unk_index`가 0보다 커야 합니다.
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

class NewsVectorizer(object):
    """ 인스턴스/레이블 어휘 사전을 생성하고 관리합니다 """
    def __init__(self, title_vocab, category_vocab):
        self.title_vocab = title_vocab
        self.category_vocab = category_vocab

    def vectorize(self, title, vector_length=-1):
        """
        매개변수:
            title (str): 공백으로 나누어진 단어 문자열
            vector_length (int): 인덱스 벡터의 길이 매개변수
        반환값:
            벡터로 변환된 제목 (numpy.array)
        """
        indices = [self.title_vocab.begin_seq_index]
        indices.extend(self.title_vocab.lookup_token(token) 
                       for token in title.split(" "))
        indices.append(self.title_vocab.end_seq_index)

        if vector_length < 0:
            vector_length = len(indices)

        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.title_vocab.mask_index

        return out_vector

    @classmethod
    def from_dataframe(cls, news_df, cutoff=25):
        """데이터셋 데이터프레임에서 Vectorizer 객체를 만듭니다
        
        매개변수:
            news_df (pandas.DataFrame): 타깃 데이터셋
            cutoff (int): Vocabulary에 포함할 빈도 임곗값
        반환값:
            NewsVectorizer 객체
        """
        category_vocab = Vocabulary()        
        for category in sorted(set(news_df.case_sort)):
            category_vocab.add_token(category)

        word_counts = Counter()
        for title in news_df.precSentences:
            for token in title.split(" "):
                if token not in string.punctuation:
                    word_counts[token] += 1
        
        title_vocab = SequenceVocabulary()
        for word, word_count in word_counts.items():
            if word_count >= cutoff:
                title_vocab.add_token(word)
        
        return cls(title_vocab, category_vocab)

    @classmethod
    def from_serializable(cls, contents):
        title_vocab = \
            SequenceVocabulary.from_serializable(contents['title_vocab'])
        category_vocab =  \
            Vocabulary.from_serializable(contents['category_vocab'])

        return cls(title_vocab=title_vocab, category_vocab=category_vocab)

    def to_serializable(self):
        return {'title_vocab': self.title_vocab.to_serializable(),
                'category_vocab': self.category_vocab.to_serializable()}

class NewsDataset(Dataset):
    def __init__(self, news_df, vectorizer):
        """
        매개변수:
            news_df (pandas.DataFrame): 데이터셋
            vectorizer (NewsVectorizer): 데이터셋에서 만든 NewsVectorizer 객체
        """
        self.news_df = news_df
        self._vectorizer = vectorizer

        # +1 if only using begin_seq, +2 if using both begin and end seq tokens
        measure_len = lambda context: len(context.split(" "))
        self._max_seq_length = max(map(measure_len, news_df.precSentences)) + 2
        

        self.train_df = self.news_df[self.news_df.split=='train']
        self.train_size = len(self.train_df)

        self.val_df = self.news_df[self.news_df.split=='val']
        self.validation_size = len(self.val_df)

        self.test_df = self.news_df[self.news_df.split=='test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')

        # 클래스 가중치
        class_counts = news_df.case_sort.value_counts().to_dict()
        def sort_key(item):
            return self._vectorizer.category_vocab.lookup_token(item[0])
        sorted_counts = sorted(class_counts.items(), key=sort_key)
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)
        
        
    @classmethod
    def load_dataset_and_make_vectorizer(cls, news_csv):
        """데이터셋을 로드하고 처음부터 새로운 Vectorizer 만들기
        
        매개변수:
            news_csv (str): 데이터셋의 위치
        반환값:
            NewsDataset의 인스턴스
        """
        news_df = pd.read_csv(news_csv)
        train_news_df = news_df[news_df.split=='train']
        return cls(news_df, NewsVectorizer.from_dataframe(train_news_df))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, news_csv, vectorizer_filepath):
        """ 데이터셋과 새로운 Vectorizer 객체를 로드합니다.
        캐시된 Vectorizer 객체를 재사용할 때 사용합니다.
        
        매개변수:
            news_csv (str): 데이터셋의 위치
            vectorizer_filepath (str): Vectorizer 객체의 저장 위치
        반환값:
            NewsDataset의 인스턴스
        """
        news_df = pd.read_csv(news_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(news_df, vectorizer)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        """파일에서 Vectorizer 객체를 로드하는 정적 메서드
        
        매개변수:
            vectorizer_filepath (str): 직렬화된 Vectorizer 객체의 위치
        반환값:
            NewsVectorizer의 인스턴스
        """
        with open(vectorizer_filepath) as fp:
            return NewsVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        """NewsVectorizer 객체를 json 형태로 디스크에 저장합니다
        
        매개변수:
            vectorizer_filepath (str): NewsVectorizer 객체의 저장 위치
        """
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self._vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        """ 벡터 변환 객체를 반환합니다 """
        return self._vectorizer

    def set_split(self, split="train"):
        """ 데이터프레임에 있는 열을 사용해 분할 세트를 선택합니다 """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """파이토치 데이터셋의 주요 진입 메서드
        
        매개변수:
            index (int): 데이터 포인트의 인덱스
        반환값:
            데이터 포인트의 특성(x_data)과 레이블(y_target)로 이루어진 딕셔너리
        """
        row = self._target_df.iloc[index]

        title_vector = \
            self._vectorizer.vectorize(row.precSentences, self._max_seq_length)

        category_index = \
            self._vectorizer.category_vocab.lookup_token(row.case_sort)

        return {'x_data': title_vector,
                'y_target': category_index}

    def get_num_batches(self, batch_size):
        """배치 크기가 주어지면 데이터셋으로 만들 수 있는 배치 개수를 반환합니다
        
        매개변수:
            batch_size (int)
        반환값:
            배치 개수
        """
        return len(self) // batch_size

def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cuda"): 
    """
    파이토치 DataLoader를 감싸고 있는 제너레이터 함수.
    걱 텐서를 지정된 장치로 이동합니다.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict

class NewsClassifier(nn.Module):
    def __init__(self, embedding_size, num_embeddings, num_channels, 
                 hidden_dim, num_classes, dropout_p, 
                 pretrained_embeddings=None, padding_idx=0):
        """
        매개변수:
            embedding_size (int): 임베딩 벡터의 크기
            num_embeddings (int): 임베딩 벡터의 개수
            num_channels (int): 합성곱 커널 개수
            hidden_dim (int): 은닉 차원 크기
            num_classes (int): 클래스 개수
            dropout_p (float): 드롭아웃 확률
            pretrained_embeddings (numpy.array): 사전에 훈련된 단어 임베딩
                기본값은 None 
            padding_idx (int): 패딩 인덱스
        """
        super(NewsClassifier, self).__init__()

        if pretrained_embeddings is None:

            self.emb = nn.Embedding(embedding_dim=embedding_size,
                                    num_embeddings=num_embeddings,
                                    padding_idx=padding_idx)        
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.emb = nn.Embedding(embedding_dim=embedding_size,
                                    num_embeddings=num_embeddings,
                                    padding_idx=padding_idx,
                                    _weight=pretrained_embeddings)
        
            
        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=embedding_size, 
                   out_channels=num_channels, kernel_size=3),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, 
                   kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, 
                   kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, 
                   kernel_size=3),
            nn.ELU()
        )

        self._dropout_p = dropout_p
        self.fc1 = nn.Linear(num_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_in, apply_softmax=False):
        """분류기의 정방향 계산
        
        매개변수:
            x_in (torch.Tensor): 입력 데이터 텐서 
                x_in.shape는 (batch, dataset._max_seq_length)입니다.
            apply_softmax (bool): 소프트맥스 활성화 함수를 위한 플래그
                크로스-엔트로피 손실을 사용하려면 False로 지정합니다
        반환값:
            결과 텐서. tensor.shape은 (batch, num_classes)입니다.
        """
        
        # 임베딩을 적용하고 특성과 채널 차원을 바꿉니다
        x_embedded = self.emb(x_in).permute(0, 2, 1)

        features = self.convnet(x_embedded)

        # 평균 값을 계산하여 부가적인 차원을 제거합니다
        remaining_size = features.size(dim=2)
        features = F.avg_pool1d(features, remaining_size).squeeze(dim=2)
        features = F.dropout(features, p=self._dropout_p)
        
        # MLP 분류기
        intermediate_vector = F.relu(F.dropout(self.fc1(features), p=self._dropout_p))
        prediction_vector = self.fc2(intermediate_vector)

        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector, dim=1)

        return prediction_vector

def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': args.model_state_file}

def update_train_state(args, model, train_state):
    """훈련 상태를 업데이트합니다.

    Components:
     - 조기 종료: 과대 적합 방지
     - 모델 체크포인트: 더 나은 모델을 저장합니다

    :param args: 메인 매개변수
    :param model: 훈련할 모델
    :param train_state: 훈련 상태를 담은 딕셔너리
    :returns:
        새로운 훈련 상태
    """

    # 적어도 한 번 모델을 저장합니다
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        train_state['stop_early'] = False

    # 성능이 향상되면 모델을 저장합니다
    elif train_state['epoch_index'] >= 1:
        loss_tm1, loss_t = train_state['val_loss'][-2:]

        # 손실이 나빠지면
        if loss_t >= train_state['early_stopping_best_val']:
            # 조기 종료 단계 업데이트
            train_state['early_stopping_step'] += 1
        # 손실이 감소하면
        else:
            # 최상의 모델 저장
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(model.state_dict(), train_state['model_filename'])

            # 조기 종료 단계 재설정
            train_state['early_stopping_step'] = 0

        # 조기 종료 여부 확인
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state

def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100

def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def get_samples():
    samples = {}
    for cat in dataset.val_df.case_sort.unique():
        samples[cat] = dataset.val_df.precSentences[dataset.val_df.case_sort==cat].tolist()[:5]
    return samples

def predict_category(title, classifier, vectorizer, max_length):
    """뉴스 제목을 기반으로 카테고리를 예측합니다
    
    매개변수:
        title (str): 원시 제목 문자열
        classifier (NewsClassifier): 훈련된 분류기 객체
        vectorizer (NewsVectorizer): 해당 Vectorizer
        max_length (int): 최대 시퀀스 길이
            노트: CNN은 입력 텐서 크기에 민감합니다. 
                 훈련 데이터처럼 동일한 크기를 갖도록 만듭니다.
    """
    vectorized_title = \
        torch.tensor(vectorizer.vectorize(title, vector_length=max_length)).to(args.device)
    result = classifier(vectorized_title.unsqueeze(0), apply_softmax=True)
    probability_values, indices = result.max(dim=1)
    predicted_category = vectorizer.category_vocab.lookup_index(indices.item())

    return {'category': predicted_category, 
            'probability': probability_values.item()}
 
if __name__ == "__main__":

   if args.expand_filepaths_to_save_dir:
      args.vectorizer_file = os.path.join(args.save_dir,
                                          args.vectorizer_file)

      args.model_state_file = os.path.join(args.save_dir,
                                          args.model_state_file)
      
      print("파일 경로: ")
      print("\t{}".format(args.vectorizer_file))
      print("\t{}".format(args.model_state_file))
      
   # CUDA 체크
   if not torch.cuda.is_available():
      args.cuda = False        
   args.device = torch.device("cuda" if args.cuda else "cpu")
   print("CUDA 사용여부: {}".format(args.cuda))

   # 재현성을 위해 시드 설정
   set_seed_everywhere(args.seed, args.cuda)

   # 디렉토리 처리
   handle_dirs(args.save_dir)

   if args.reload_from_files:
      # 체크포인트를 로드합니다.
      dataset = NewsDataset.load_dataset_and_load_vectorizer(args.news_csv,
                                                               args.vectorizer_file)
   else:
      # 데이터셋과 Vectorizer를 만듭니다.
      final_reviews_maker(args)
      dataset = NewsDataset.load_dataset_and_make_vectorizer(args.news_csv)
      dataset.save_vectorizer(args.vectorizer_file)

   vectorizer = dataset.get_vectorizer()

   # w2v를 사용하거나 랜덤하게 임베딩을 초기화합니다
   if args.use_w2v:
      embeddings = make_embedding_matrix(w2v_filepath=args.w2v_filepath, vectorizer=vectorizer)
      print("사전 훈련된 임베딩을 사용합니다")
   else:
      print("사전 훈련된 임베딩을 사용하지 않습니다")
      embeddings = None

   classifier = NewsClassifier(embedding_size=args.embedding_size, 
                              num_embeddings=len(vectorizer.title_vocab),
                              num_channels=args.num_channels,
                              hidden_dim=args.hidden_dim, 
                              num_classes=len(vectorizer.category_vocab), 
                              dropout_p=args.dropout_p,
                              pretrained_embeddings = embeddings, 
                              padding_idx=0)

   # weights = torch.FloatTensor(model.wv.vectors)
   # embedding = nn.Embedding.from_pretrained(weights)

   dataset.class_weights = dataset.class_weights.to(args.device)
      
   loss_func = nn.CrossEntropyLoss(dataset.class_weights)
   optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
   scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                          mode='min', factor=0.5,
                                          patience=1)

   train_state = make_train_state(args)

   epoch_bar = tqdm.std.tqdm(desc='training routine', 
                           total=args.num_epochs,
                           position=0)
   dataset.set_split('train')
   train_bar = tqdm.std.tqdm(desc='split=train',
                           total=dataset.get_num_batches(args.batch_size), 
                           position=1, 
                           leave=True)
   dataset.set_split('val')
   val_bar = tqdm.std.tqdm(desc='split=val',
                           total=dataset.get_num_batches(args.batch_size), 
                           position=1, 
                           leave=True)

   if args.finalTest == False :

      classifier = classifier.to(args.device)
      try:
            for epoch_index in range(args.num_epochs):
               train_state['epoch_index'] = epoch_index

               # 훈련 세트에 대한 순회

               # 훈련 세트와 배치 제너레이터 준비, 손실과 정확도를 0으로 설정
               dataset.set_split('train')
               batch_generator = generate_batches(dataset, 
                                                batch_size=args.batch_size, 
                                                device=args.device)
               running_loss = 0.0
               running_acc = 0.0
               classifier.train()

               for batch_index, batch_dict in enumerate(batch_generator):
                  # 훈련 과정은 5단계로 이루어집니다

                  # --------------------------------------
                  # 단계 1. 그레이디언트를 0으로 초기화합니다
                  optimizer.zero_grad()

                  # 단계 2. 출력을 계산합니다
                  y_pred = classifier(batch_dict['x_data'])

                  # 단계 3. 손실을 계산합니다
                  loss = loss_func(y_pred, batch_dict['y_target'])
                  loss_t = loss.item()
                  running_loss += (loss_t - running_loss) / (batch_index + 1)

                  # 단계 4. 손실을 사용해 그레이디언트를 계산합니다
                  loss.backward()

                  # 단계 5. 옵티마이저로 가중치를 업데이트합니다
                  optimizer.step()
                  # -----------------------------------------
                  
                  # 정확도를 계산합니다
                  acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
                  running_acc += (acc_t - running_acc) / (batch_index + 1)

                  # 진행 상태 막대 업데이트
                  train_bar.set_postfix(loss=running_loss, acc=running_acc, 
                                       epoch=epoch_index)
                  train_bar.update()

               train_state['train_loss'].append(running_loss)
               train_state['train_acc'].append(running_acc)

               # 검증 세트에 대한 순회

               # 검증 세트와 배치 제너레이터 준비, 손실과 정확도를 0으로 설정
               dataset.set_split('val')
               batch_generator = generate_batches(dataset, 
                                                batch_size=args.batch_size, 
                                                device=args.device)
               running_loss = 0.
               running_acc = 0.
               classifier.eval()

               for batch_index, batch_dict in enumerate(batch_generator):

                  # 단계 1. 출력을 계산합니다
                  y_pred =  classifier(batch_dict['x_data'])

                  # 단계 2. 손실을 계산합니다
                  loss = loss_func(y_pred, batch_dict['y_target'])
                  loss_t = loss.item()
                  running_loss += (loss_t - running_loss) / (batch_index + 1)

                  # 단계 3. 정확도를 계산합니다
                  acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
                  running_acc += (acc_t - running_acc) / (batch_index + 1)
                  val_bar.set_postfix(loss=running_loss, acc=running_acc, 
                                    epoch=epoch_index)
                  val_bar.update()

               train_state['val_loss'].append(running_loss)
               train_state['val_acc'].append(running_acc)

               train_state = update_train_state(args=args, model=classifier,
                                                train_state=train_state)

               scheduler.step(train_state['val_loss'][-1])

               if train_state['stop_early']:
                  break

               train_bar.n = 0
               val_bar.n = 0
               epoch_bar.update()
               
      except KeyboardInterrupt:
            print("Exiting loop")

   # 가장 좋은 모델을 사용해 테스트 세트의 손실과 정확도를 계산합니다
   classifier.load_state_dict(torch.load(train_state['model_filename']))

   classifier = classifier.to(args.device)
   dataset.class_weights = dataset.class_weights.to(args.device)
   loss_func = nn.CrossEntropyLoss(dataset.class_weights)

   dataset.set_split('test')
   batch_generator = generate_batches(dataset, 
                                    batch_size=args.batch_size, 
                                    device=args.device)
   running_loss = 0.
   running_acc = 0.
   classifier.eval()

   for batch_index, batch_dict in enumerate(batch_generator):
         # 출력을 계산합니다
         y_pred =  classifier(batch_dict['x_data'])
         
         # 손실을 계산합니다
         loss = loss_func(y_pred, batch_dict['y_target'])
         loss_t = loss.item()
         running_loss += (loss_t - running_loss) / (batch_index + 1)

         # 정확도를 계산합니다
         acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
         running_acc += (acc_t - running_acc) / (batch_index + 1)

   train_state['test_loss'] = running_loss
   train_state['test_acc'] = running_acc

   print("테스트 손실: {};".format(train_state['test_loss']))
   print("테스트 정확도: {}".format(train_state['test_acc']))

   val_samples = get_samples()

   #title = input("Enter a news title to classify: ")
   classifier = classifier.to("cpu")

   for truth, sample_group in val_samples.items():
         print(f"True Category: {truth}")
         print("="*30)
         for sample in sample_group:
            prediction = predict_category(sample, classifier, 
                                       vectorizer, dataset._max_seq_length + 1)
            print("예측: {} (p={:0.2f})".format(prediction['category'],
                                                   prediction['probability']))
            print("\t + 샘플: {}".format(sample))
         print("-"*30 + "\n")

   # 가장 좋은 모델을 사용해 테스트 세트의 손실과 정확도를 계산합니다
   print("가장 좋은 모델을 사용해 테스트 세트의 손실과 정확도를 계산합니다")
   classifier.load_state_dict(torch.load(train_state['model_filename'], map_location=args.device))
   classifier = classifier.to(args.device)
   classifier.eval()
   test_review_01 = """
   물권행동 카라 제공
[법률방송뉴스] 길고양이 등을 잔인하게 학대하는 사진과 영상을 오픈채팅에 공유한 이른바 ‘동물판 n번방’에서 행동대장으로 불린 이모씨에게 집행유예가 내려지며 ‘솜방망이 처벌’이라는 비판을 받고 있습니다. 

동물단체에 따르면 대전지방법원 서산지원 재판부는 오늘(11일) 이씨에게 징역 4개월에 집행유예 2년, 벌금 100만원을 선고했습니다.

이씨는 지난 1월 검정색 길고양이의 허리를 화살로 관통하고 피투성이가 된 채 바닥에 누워있는 사진을 오픈채팅에 공유해 동물보호법 위반 혐의로 재판에 넘겨졌습니다. 

당시 이씨는 채팅방에서 ‘멀리서 쏴서 빗나갔는데 척추에 맞아서 후지가 마비돼서 운 좋게 잡을 수 있었다’고 언급한 것으로 알려졌습니다. 이후 ‘고양이를 잔혹하게 학대하고 먹는 단체 오픈카톡방을 수사하고 처벌해달라'는 제목의 청와대 국민청원 게시판에는 27만명 이상이 동의했습니다.

재판에 앞서 동물권단체 카라는 오늘(11일) 오후 1시 대전지방법원 서산지원 앞에서 이모씨의 법정 최고형 선고를 촉구하는 기자회견을 열었습니다.

최민경 카라 활동가는 “검은 고양이의 눈은 최대한으로 확장돼 카메라를 바라보고 있었는데 공포와 고통이 뒤섞인 모습이었음을 누구나 확인할 수 있었다”며 “그는 불법사냥과 동물학대를 마치 게임을 즐기는 듯이 채팅방에 언급하고 사진과 동영상을 수시로 공유해왔다”고 말했습니다.

이어 “검찰은 이 무차별한 동물학대범에 동물보호법 법정 최고형인 3년 징역을 구형했다”며 “그러나 법정에서 이씨는 반성조차 하지 않고 ‘고통없이 동물을 죽였을 뿐’이라고 뻔뻔하게 진술했다”고 밝혔습니다. 또 “현행 동물보호법상 최고형이 징역 3년인 만큼 오늘 서산지원 재판부에서 부디 최고형 선고를 내려달라”고 요청했습니다.



자신을 서산 동물권행동 활동가라고 밝힌 시민 A씨는 “1차 공판 때도 참여했다. 다시 나온 이유는 이씨의 태도에 분노심을 느꼈기 때문”이라며 “일반 국민들이 동물에 대해 어떤 생각을 가지는지 보여줄 필요성이 있다”고 했습니다.

경기 군포시에서 길고양이를 구조하고 치료하는 활동을 하고 있는 시민 B씨는 “오늘 (이씨가) 법정구속 된다면 역사적인 날이 될 것 같다. 죽어간 아이들의 원한을 갚아주는 현장이 보고 싶어 서산까지 왔다”며 “생명 해치는 사람들이 행위를 못하게 하는 건 사법부의 의무”라고 밝혔습니다.

서울에서 온 시민 C씨는 “길고양이 학대 살해에 대한 수많은 청원이 있었지만 국민청원 답변요건 20만을 넘은 적이 단 한 번도 없었는데 답변요건을 충족한 첫 사례”라며 “그만큼 많은 시민을 분노하게 한 사건이므로 실형 선고가 매우 중요하다”고 말했습니다. 이어 “유영철, 강호순, 이영학 등 연쇄살인범 상당수가 가학적 동물학대 전력이 있다”며 “한국에서도 범죄예방 차원에서 국가가 관리할 수 있는 시스템이 마련됐으면 한다”고 밝혔습니다.

하지만 재판부가 이씨에게 집행유예를 선고하자, 카라는 긴급 기자회견을 열어 오픈채팅 고어방 최종판결에 대해 규탄했습니다.

최민경 활동가는 “말도 안 되는 결과에 이 상황을 도저히 받아들일 수 없다”며 한숨을 내쉬었습니다. 그러면서 “약자를 위한 대한민국 재판부는 어디에 있냐”며 목소리를 높였습니다. 

이어 “시민들은 법정에서 엉엉 울면서 나왔다. 하지만 저 범죄자는 두발로 당당히 걸어나갔다”면서 “말 못하는 입장의 동물들은 어떻게 보호받아야 하냐”고 울부짖었습니다. 그러면서 “카라와 시민들은 오늘의 선고를 절대 잊지 않을 것”이라며 기자회견을 마무리했습니다.

한편 동물 학대 사건은 온라인 매체를 통해 빈번히 발생하고 있을 뿐만 아니라 수법 역시 다양해지고 있습니다. 이에 동물보호법 위반에 대한 처벌은 솜방망이 수준에 그친다는 지적과 함께 동물 학대 예방을 위한 실질적 대책 마련이 시급하다는 목소리가 나오고 있습니다.

출처 : 법률방송뉴스(http://www.ltn.kr) """

   test_review_02="""  [이혼등ㆍ이혼등청구의소] 항소[각공2021상,53]

【판시사항】

법률상 부부인 갑과 을이 혼인기간 동안 누적된 불만과 갈등을 이유로 상호 이혼 등의 본소 및 반소를 제기한 사안에서, 갑과 을의 혼인관계는 혼인기간 동안 상호 간에 누적된 불만과 갈등에 더하여 갑의 협의이혼 숙려기간 중의 교제 등 갑과 다른 이성과의 부적절한 관계가 주요한 원인이 되어 파탄에 이르게 되었다고 보이므로, 유책배우자인 갑의 이혼 및 위자료 청구를 배척하고, 을의 이혼 및 위자료 청구를 인용한 사례

【판결요지】
법률상 부부인 갑과 을이 혼인기간 동안 누적된 불만과 갈등을 이유로 상호 이혼 등의 본소 및 반소를 제기한 사안이다.

갑이 협의이혼의사확인 신청일 이후 숙려기간에 다른 이성과 교제한 점, 일반적으로 부부간 갈등과정에서 별거기간 또는 협의이혼 숙려기간은 혼인관계 유지 등에 관한 진지한 고민의 시간이자 혼인관계 회복을 위한 노력의 시간이기도 하므로 특별한 사정이 없는 한 협의이혼 숙려기간 중 다른 이성과 교제하는 것 역시 혼인관계의 유지를 방해하고 상대방의 신뢰를 훼손하는 부정행위에 해당하는데, 갑이 다른 이성과 교제를 시작한 시기, 갑과 을 사이의 갈등이 증폭된 경위와 그 시기 등에 비추어 보면, 갑과 을의 혼인관계는 혼인기간 동안 상호 간에 누적된 불만과 갈등에 더하여 갑의 협의이혼 숙려기간 중의 교제 등 갑과 다른 이성과의 부적절한 관계가 주요한 원인이 되어 파탄에 이르게 되었다고 보이므로, 유책배우자인 갑의 이혼 및 위자료 청구를 배척하고, 을의 이혼 및 위자료 청구를 인용한 사례이다.

【참조조문】

민법 제806조, 제840조 제1호, 제6호, 제843조



【전 문】

【원고(반소피고)】 원고(반소피고) (소송대리인 법무법인 해람 담당변호사 소현완 외 1인)

【피고(반소원고)】 피고(반소원고)(소송대리인 법무법인 로앤 담당변호사 이명조 외 1인)

【사건본인】 사건본인 1 외 1인

【변론종결】 2019. 10. 18.

【주 문】

1. 반소에 의하여, 원고(반소피고)와 피고(반소원고)는 이혼한다.

2. 원고(반소피고)는 피고(반소원고)에게 위자료 10,000,000원과 이에 대하여 2019. 8. 8.부터 2020. 2. 14.까지는 연 5%, 그 다음 날부터 다 갚는 날까지는 연 12%의 각 비율로 계산한 돈을 지급하라.

3. 원고(반소피고)의 본소 이혼 및 위자료 청구와 피고(반소원고)의 반소 나머지 위자료 청구를 각 기각한다.

4. 피고(반소원고)는 원고(반소피고)에게 재산분할로 70,000,000원과 이에 대하여 이 판결 확정일 다음 날부터 다 갚는 날까지 연 5%의 비율로 계산한 돈을 지급하라.

5. 사건본인들의 친권자 및 양육자로 피고(반소원고)를 지정한다.

6. 원고(반소피고)는 피고(반소원고)에게

가. 사건본인들의 과거양육비로 15,000,000원을 지급하고,

나. 사건본인들의 장래양육비로 2020. 2.부터 사건본인들이 각 성년에 이를 때까지 사건본인 1인당 월 50만 원을 매월 말일에 지급하라.

7. 원고(반소피고)는 다음과 같이 사건본인들과 면접교섭할 수 있다.

가. 일정

1) 월 2회: 매월 둘째, 넷째 토요일 10:00부터 일요일 19:00까지(1박 2일)

2) 그 밖에 원고(반소피고)와 피고(반소원고)는 만나기 2일 전까지 면접교섭의 가능 여부, 시간과 장소, 인도방법 등을 협의하고, 그 협의에 따라 위 일정을 변경하거나 위 일정 외에 추가로 면접교섭을 실시할 수 있다.

3) 원고(반소피고)와 피고(반소원고)는 면접교섭을 계획하고 실시함에 있어 사건본인들의 정서 상태와 건강 등을 우선적으로 고려하여야 한다.

나. 장소: 원고(반소피고)가 책임질 수 있는 장소

다. 인도방법: 원고(반소피고)가 사건본인들의 주거지 또는 피고(반소원고)와 협의하여 정한 장소로 가서 사건본인들을 인도받고, 면접교섭을 마친 후에는 다시 같은 장소로 데려다주면서 인도하는 방법

라. 허용의무: 피고(반소원고)는 위와 같은 면접교섭이 원만하게 실시될 수 있도록 적극 협조하여야 하고, 이를 방해하여서는 아니 된다.

8. 소송비용은 본소, 반소를 통틀어 각자 부담한다.

9. 제2, 6항은 가집행할 수 있다.

【청구취지】 ○ 본소: 별지 청구취지 기재와 같다.

○ 반소: 주문 제1, 5항, 원고(반소피고, 이하 ‘원고’라 한다)는 피고(반소원고, 이하 ‘피고’라 한다)에게 위자료 3,000만 원과 이에 대하여 이 사건 반소장 부본 송달일 다음 날부터 다 갚는 날까지 연 12%의 비율로 계산한 돈을, 재산분할로 21,778,136원과 이에 대하여 이 판결 확정일 다음 날부터 다 갚는 날까지 연 5%의 비율로 계산한 돈을 각 지급하라. 원고는 피고에게 사건본인들의 과거양육비로 3,600만 원을 지급하고, 장래양육비로 이 판결 확정일로부터 사건본인들이 각 성년이 이를 때까지 사건본인 1인당 월 100만 원을 매월 말일에 지급하라.

【이 유】

1. 인정 사실

가. 원고와 피고는 2006. 8. 22. 혼인신고를 마치고 그 사이에 사건본인들을 둔 법률상 부부이다.

나. 피고는 혼인기간 동안 건설업을 영위하면서 소득을 올렸고, 원고는 전업주부로 가사와 자녀 양육을 전담하다가 2011. 9.경부터 초등학교 방과 후 교사로 근무하였다.

다. 원고는 2013년경 피고와 함께 골프 동호회에 가입하여 활동하였는데, 그 무렵 같은 동호회 남자회원으로부터 골프의류를 대신 구매해달라는 요청을 받고 이를 구입하여 원고의 차량 뒷좌석에 놓아두었다가 이를 발견한 피고로부터 부정행위 의심을 받게 되었다. 위 사건을 계기로 원고와 피고는 동호회 활동을 그만두었고, 피고는 원고의 외부활동을 경계하면서 통제하려 하였다.

라. 원고는 2016. 9.경 피고의 동의를 받고 배드민턴 동호회 활동을 다시 시작하였으나, 원고가 약속한 귀가시간인 밤 10시를 넘기는 경우가 잦아지고 방학을 맞아 낮에도 배드민턴 운동을 하게 되자, 이에 불만을 가진 피고가 2018. 1.경 동호회 활동을 그만둘 것을 요구하였다. 원고는 이를 거절하면서 오히려 이혼을 요구하였고, 피고가 이에 응하면서 원고와 피고는 2018. 2. 14.경 이 법원에 협의이혼의사확인 신청을 하였다.

마. 한편 원고는 2018. 1. 말 또는 2018. 2. 초경부터 배드민턴 동호회 회장이던 소외 1을 개인적으로 만나기 시작하였다. 원고는 2018. 3. 4.경 소외 1을 만나 영화를 관람하였고, 2018. 3. 13.경에는 함께 피아노 공연을 보기도 하였으며, 피고와의 이혼 등 법률문제 상담을 위해 변호사 사무실을 찾을 때 소외 1과 동행하였다.

바. 피고는 원고의 SNS 계정에 몰래 접속하여 원고와 소외 1의 관계를 알게 되었고, 2018. 3. 22.경에는 배드민턴 동호회 회원들을 단체 카톡방에 초대하여 원고와 소외 1이 부정행위를 하였다는 취지의 글을 올렸다. 이에 원고와 소외 1이 피고를 명예훼손 등의 혐의로 고발하였는데, 원고에 대한 혐의는 기소유예처분이, 소외 1에 대한 혐의는 벌금 100만 원의 약식명령이 내려졌다.

사. 원고는 2018. 4. 4.경 집을 나와 그 무렵부터 피고와 별거하고 있다.

[인정 근거] 갑 제1 내지 4, 9, 11호증(가지번호 있는 것은 가지번호 포함, 이하 같다), 을 제6, 18, 20호증의 각 기재 또는 영상, 가사조사관 작성의 조사보고서, 변론 전체의 취지

2. 본소 및 반소의 각 이혼 및 위자료 청구에 관한 판단

가. 본소 및 반소 이혼 청구: 본소 기각(유책배우자), 반소 인용(민법 제840조 제1, 6호)

나. 본소 및 반소 위자료 청구: 본소 기각, 반소 일부 인용(위자료 1,000만 원 및 이에 대한 지연손해금)

[판단 근거]

○ 혼인관계 파탄 인정: 위 인정 사실에 나타난 부부간 갈등의 내용 및 정도, 원고와 피고가 2018. 4.경부터 별거 중이고, 이 사건 본소와 반소로 이혼을 구하고 있는 점 등 변론에 나타난 여러 사정을 참작

○ 혼인파탄의 주된 책임: 원고와 소외 1이 교제하기 시작한 정확한 날짜를 특정할 수는 없으나 적어도 협의이혼의사확인 신청일 이후 숙려기간에 교제한 점에 대하여는 원고도 인정하고 있다. 일반적으로 부부간 갈등과정에서 별거기간 또는 협의이혼 숙려기간은 혼인관계 유지 등에 관한 진지한 고민의 시간이자 혼인관계 회복을 위한 노력의 시간이기도 하므로 특별한 사정이 없는 한 협의이혼 숙려기간 중 다른 이성과 교제하는 것 역시 혼인관계의 유지를 방해하고 상대방의 신뢰를 훼손하는 부정행위에 해당한다. 나아가 원고와 소외 1이 교제를 시작한 시기, 원고와 피고 사이의 갈등이 증폭된 경위와 그 시기 등에 비추어 보면, 원고와 소외 1의 관계가 이 사건 혼인관계 파탄에 상당한 영향을 미친 것으로 보인다. 한편 피고에게도 혼인기간 동안 원고의 입장을 이해하고 서로의 입장 차이를 조율하려는 적극적인 노력을 기울이기보다는 원고를 비난하고 통제하려는 가부장적인 방법으로 갈등을 무마하려 한 잘못이 있으나, 그 책임의 정도가 원고의 책임을 상쇄할 정도에 이른다고 보이지는 않는다. 따라서 여러 사정을 종합해 보면, 원고와 피고의 혼인관계는 혼인기간 동안 상호 간에 누적된 불만과 갈등에 더하여 원고와 소외 1의 부적절한 관계가 주요한 원인이 되어 파탄에 이르게 되었다고 봄이 상당하므로, 그 책임은 원고에게 조금 더 있다고 판단된다.

○ 위자료 액수: 혼인관계 파탄의 경위 및 책임의 정도, 원고와 피고의 혼인기간, 나이, 직업, 경제력 등 변론에 나타난 여러 사정을 참작하여 원고가 피고에게 지급하여야 할 위자료의 액수를 1,000만 원으로 정한다.

다. 소결론

1) 반소에 의하여, 원고와 피고는 이혼한다.

2) 유책배우자인 원고의 이혼 및 위자료 청구는 이유 없어 기각한다.

3) 원고는 피고에게 위자료로 1,000만 원과 이에 대하여 이 사건 반소장 부본 송달일 다음 날인 2019. 8. 8.부터 원고가 그 이행의무의 존부 및 범위에 관해 항쟁함이 상당한 이 판결 선고일인 2020. 2. 14.까지는 민법이 정한 연 5%, 그 다음 날부터 다 갚는 날까지는 소송촉진 등에 관한 특례법이 정한 연 12%의 각 비율로 계산한 지연손해금을 지급할 의무가 있다.

3. 본소 및 반소의 각 재산분할 청구에 관한 판단

가. 분할대상재산 및 가액: 별지 분할재산명세표 기재와 같다(분할대상재산 및 가액을 산정하는 기준시점은 원칙적으로 이 사건 변론종결일로 정하되, 소비나 은닉이 용이한 금융재산은 원고와 피고가 별거하기 시작한 날로서 혼인관계가 파탄된 시점으로 볼 수 있는 2018. 4. 4.경을 기준으로 하고, 그 밖에 원고와 피고가 일치하여 진술하거나 금융거래정보제출명령 결과 등에 나타난 가액에 대하여는 특별한 사정이 없는 한 그대로 인정한다. 다만 가액이 10,000원 미만인 것은 제외한다).

나. 불포함재산

1) 원고의 ○○은행 예금채권: 피고는 원고의 ○○은행 예금채권이 15,068,245원에 이른다고 주장하나, 갑 제20호증의 기재, 이 법원의 ○○은행 및 △△△△△△에 대한 각 금융거래제출명령 결과에 변론 전체의 취지를 종합하면, 원고가 혼인 파탄일을 전후하여 해지한 각종 보험 및 펀드 해지환급금 등 합계 1,500만 원 상당이 원고의 현 거주지 전세보증금 중 일부에 충당된 사실이 인정되므로, 전세보증금 1억 4,000만 원을 원고의 적극재산에 포함시킨 이상 위 1,500만 원 상당을 중복하여 원고의 적극재산에 반영할 수는 없다. 피고의 이 부분 주장은 이유 없다.

2) 피고의 어머니 소외 2에 대한 1억 원의 차용금 채무: 피고는 2017. 6.경부터 2018. 5.까지 (주)현창건설로부터 부산 강서구 (주소 생략) 소재 □□□ 빌딩 및 ◇◇빌딩의 신축공사를 도급받았는데, 2017. 8. 31. 인부들에게 지급할 2017. 7.분 노무비로 약 9,000만 원 상당이 필요하였으나 지급받은 기성금이 약 6,700만 원밖에 되지 않아 자신의 어머니 소외 2로부터 1억 원을 빌려 이에 충당하였다는 취지로 주장하나, 피고 명의 ☆☆은행 계좌(계좌번호 생략)에 소외 2 명의로 입금된 금액은 1억 원이 아닌 9,400만 원인 점, 피고 주장에 의하더라도 부족한 노무비 차액은 2,300만 원에 불과하고 나머지 돈의 용처에 관하여는 아무런 주장ㆍ입증이 없는 점, 피고가 그동안 소외 2에게 월 50만 원씩의 이자를 지급한 내역을 찾을 수 없는 점 등에 비추어 보면, 을 제15, 16호증의 각 기재만으로는 피고가 소외 2로부터 1억 원을 차용하였고 위 돈이 혼인공동생활을 위해 사용되었음을 인정하기에 부족하고 달리 이를 인정할 증거가 없다. 피고의 이 부분 주장은 이유 없다.

3) 피고의 2018. 3. 23.자 6,900만 원 및 2018. 3. 27.자 3,500만 원의 각 대출채무: 피고는, 2017. 6. 20.경 피고가 (주)현창건설로부터 도급받은 공사 중 일부를 소외 3에게 하도급을 주었는데, 2018. 3. 23. ☆☆은행에서 6,900만 원을 대출받아 소외 3에게 하도급대금 7,000만 원을 지급하였고, 2018. 3. 27. 같은 은행에서 3,500만 원을 대출받아 자재업자인 소외 4에게 자재대금 3,500만 원을 지급하였는바, 이는 혼인파탄일인 2018. 4. 4. 이전에 공사일정에 따라 정상적으로 발생한 채무로서 분할대상에 포함되어야 한다고 주장한다. 그러나 갑 제23호증의 기재, 이 법원의 (주)☆☆은행에 대한 금융거래제출명령 결과에 변론 전체의 취지를 종합하여 알 수 있는 다음과 같은 사정, 즉 피고는 2017. 7.경부터 2018. 9.경까지 (주)현창건설로부터 수시로 기성금을 지급받았음에도 소외 3에게는 위 7,000만 원 외에 기성금을 지급한 내역이 없는 점, 소외 3은 2018. 6. 8. 피고 명의 ☆☆은행 계좌(계좌번호 생략)로 7,000만 원을 송금하여 위 7,000만 원을 반환한 듯한 내역이 있는 점, 위 소외 4 또한 2018. 11. 30. 같은 계좌로 3,500만 원을 송금하여 위 3,500만 원을 반환한 듯한 내역이 있는 점, 피고는 소외 3과 소외 4가 위와 같이 피고로부터 지급받은 돈을 몇 달 뒤에 돌려준 이유에 관하여 아무런 설명을 하지 않고 있는 점 등에 비추어 보면, 을 제23호증의 기재만으로는 위 각 대출채무가 피고의 사업상 불가피한 대출이었고, 나아가 위 대출금이 혼인공동생활을 위해 사용되었음을 인정하기에 부족하고, 달리 이를 인정할 증거가 없다. 피고의 이 부분 주장은 이유 없다.

4) 피고의 ▽▽▽▽보험(어린이보험) 예상환급금 2건 및 ◎◎◎◎◎◎◎보험 예상환급금 2건: 이 법원의 ▽▽▽▽보험 주식회사 및 ◎◎◎◎◎◎◎보험 주식회사에 대한 각 금융거래정보제출명령 결과에 의하면, 위 각 보험계약은 혼인파탄일을 전후하여 체결된 것으로서 원고의 기여도를 인정하기 어렵다. 피고의 적극재산에서 제외한다.

다. 재산분할의 비율과 방법

1) 재산분할의 비율: 원고 40%, 피고 60%

[판단 근거] 원피고의 나이, 직업, 혼인생활의 과정과 기간, 분할대상재산의 형성 및 유지에 대한 원피고의 기여 정도, 아래에서 보는 바와 같이 피고가 사건본인들을 양육하게 되는 점 등 변론에 나타난 여러 사정을 참작

2) 재산분할의 방법: 분할대상재산의 명의와 형태, 취득경위, 분할의 편의성, 현재의 이용상황 등을 고려하여 원고와 피고 명의로 된 적극재산 및 소극재산은 그 명의대로 각자에게 귀속시키기로 하되, 위 분할비율에 따라 원고에게 궁극적으로 귀속되어야 할 금액 중 부족한 부분을 피고가 원고에게 현금으로 지급하는 것으로 정함

3) 재산분할정산금: 70,000,000원(아래에서 계산된 차액을 하회하는 금액으로 정함)

[계산식]

① 원고와 피고의 순재산 중 재산분할 비율에 따른 원고의 몫: 105,306,767원

(= 원고와 피고의 순재산 합계 263,266,919원 × 40%, 원 미만은 버림)

② 원고의 순재산과 위 ①항 기재 금액과의 차액: 70,577,879원

(= 105,306,767원 - 34,728,888원)

라. 소결론

따라서 피고는 원고에게 재산분할로 70,000,000원과 이에 대하여 이 판결 확정일 다음 날부터 다 갚는 날까지 민법이 정한 연 5%의 비율로 계산한 지연손해금을 지급할 의무가 있다.

4. 본소 및 반소 각 친권자ㆍ양육자 지정 및 양육비 등 청구에 관한 판단

가. 친권자 및 양육자: 피고로 지정

[판단 근거] 사건본인들의 나이와 성별, 현재의 양육 상황, 원고와 피고의 양육 환경 및 당사자들의 의사 등 변론에 나타난 여러 사정을 참작

나. 양육비: 사건본인들의 나이 및 양육 상황, 원고와 피고의 나이, 직업, 소득 및 생활 능력, 서울가정법원이 공표한 양육비산정기준표, 당사자의 의사 등 변론에 나타난 여러 사정을 참작하여, 사건본인들의 양육비를 1인당 월 50만 원으로 정한다. 따라서 원고는 피고에게 사건본인들의 과거양육비로 1,500만 원(2018. 4.부터 2020. 1.까지 월 100만 원으로 계산한 미지급 양육비 중 일부)을 지급하고, 장래양육비로 2020. 2.부터 사건본인들이 각 성년에 이를 때까지 사건본인 1인당 월 50만 원을 매월 말일에 지급할 의무가 있다.

다. 면접교섭: 비양육친은 사건본인의 복리에 반하지 않는 한 사건본인과 면접교섭할 권리가 있는바, 사건본인들의 나이, 양육 상황, 면접교섭에 관한 당사자의 의사 등을 종합하여, 주문 제7항 기재와 같이 면접교섭에 관하여 정한다.

5. 결론

그렇다면 피고의 반소 이혼 청구는 이유 있어 인용하고, 반소 위자료 청구는 위 인정 범위 내에서 이유 있어 인용하고 나머지 청구는 이유 없어 기각하며, 원고의 본소 이혼 및 위자료 청구는 이유 없어 기각하고, 재산분할, 친권자 및 양육자 지정, 양육비, 면접교섭 청구에 관하여는 위와 같이 정하기로 하여, 주문과 같이 판결한다."""

   test_review_03 = """
    외교부는 전체 직원의 절반 이상이 전 세계 167곳 재외공관에 흩어져 근무하는 특수한 부처입니다.

이 때문에 다른 부처와는 구별되는 몇몇 관행들을 갖고 있는데, 그 중 하나가 '경조사 계좌'입니다.

직원들 편의 차원에서 경조사비를 외교부 명의 계좌로 접수한 뒤, 경조사 당사자에게 전달해주는 내부 서비스가 10년 넘게 운영돼 왔습니다.

그런데 지난해 사건이 터졌습니다.

이 경조사 계좌를 관리하던 직원이 경조사비를 횡령한 혐의로 해고 당하고, 검찰에 고발된 겁니다. 사건 발생 1달 뒤 이 같은 내용이 언론에 보도되기도 했습니다.

그로부터 1년. KBS가 최근 법원 판결문 열람을 통해 확인한 결과 이 직원은 올해 8월 1심에서 유죄 선고를 받은 것으로 확인됐습니다.

사건의 전모가 요약돼 있는 판결문 내용과 사건의 뒷이야기를 살펴봤습니다.

■ 1심 법원 "전액 횡령 아니고 상당 부분 변제했지만…실형 불가피"

판결문을 보면, 직원 A 씨는 2018년 9월 초부터 지난해 9월 중순까지 외교부 운영지원담당관실 실무관으로 일하며 경조사 계좌 관리를 담당했습니다. 운영지원담당관실은 외교부 자금 운용·회계와 결산, 소속 공무원의 급여·복리후생, 청사 관리, 외교행낭 등에 관한 사무를 맡는 곳입니다.

범행은 2019년 8월 중순 처음 시작됐습니다. 경조사 계좌에 입금돼 있던 50만 원을 A 씨가 본인 계좌로 이체한 뒤, 생활비 등 개인적인 용도로 사용한 것입니다.

그때부터 이듬해 9월 중순까지 13개월 동안, A 씨가 무려 117차례에 걸쳐 2억 180만 원을 개인 용도로 써 돈을 횡령했다는 게 검찰 수사 결과였습니다.

그는 올해 3월 업무상 횡령 혐의로 기소됐는데, 재판에서 혐의를 인정한 것으로 알려졌습니다. 이에 한 차례의 재판 이후 바로 선고기일이 잡혔습니다.

담당 재판부였던 서울북부지방법원 형사7단독(판사 나우상)은 올해 8월 A 씨에게 징역 1년 2개월의 실형을 선고했습니다.

재판부는 공소사실을 유죄로 인정하면서도 A 씨가 실제 2억 원 전체를 개인적 용도로 소비한 건 아니라고 판단했습니다.

돈이 필요할 때 경조사 계좌에서 돈을 빼서 썼다가 다시 채워놓는 등, 횡령 기간 동안 1억 1,340만 원은 계좌에 다시 돌려놨다는 것입니다.

결국 개인적으로 쓴 돈은 8,840만 원 정도가 되는 건데 이 중 4,300만 원을 변제한 점, 부양해야 할 가족이 많은 점 등도 A 씨에겐 유리한 정상이라고 재판부는 밝혔습니다.

그러나 A 씨가 오랜 기간에 걸쳐 횡령한 금액이 적지 않고, 일부 횡령금을 도박에 사용한 점 등을 참작하면 실형이 불가피하다고 했습니다.

재판부는 다만 A 씨가 피해 금액을 외교부에 변제하기 위해선 경제 활동을 할 필요가 있다는 이유로, 법정 구속을 하진 않았습니다.

A 씨는 판결 선고 당일 즉각 항소했습니다. 항소심 재판은 아직 시작되지 않았습니다.

■ 외교부, 12년 만에 경조사 계좌 폐지…뒤늦은 '주인 찾기'는 난항

지난해 횡령 사건 직후, 외교부는 후속 조치로 곧장 경조사 계좌를 폐지했습니다. 계좌를 개설한 지 12년 만이었습니다.

외교부 관계자는 "필요하면 이제 경조사비를 직접 당사자 계좌로 입금하라고 공지했고, 당사자 계좌를 공지하는 방식으로 변경했다"고 설명했습니다.

경조사가 생겼다는 연락이 오면 운영지원담당관실에서 당사자에게 계좌번호를 문의하고, 당사자가 희망한다고 하면 계좌번호를 경조사 소식과 함께 내부 게시판에 공지해주는 방식입니다.

이 관계자는 "경조사를 당한 입장에서 본인 계좌를 공지한다는 건 '돈을 내라'는 건데, 그걸 민망해하시는 분들이 계시다"면서 "공지를 안 하는 경우가 많아지고 그러다보면 계좌를 모르니 밖(재외공관)에 있는 분들은 어떡하지 하다가 (축의·부의를) 못하게 되기도 한다. 마음을 전하는 게 원활한 상황은 아니다"라고 분위기를 전했습니다.

횡령 금액의 변제·반환도 남은 과제이지만, A 씨가 변제를 마치더라도 그 돈의 주인을 정확히 찾기는 어려운 상황인 것으로 전해졌습니다.

접수된 모든 경조사비를 한 통장에서 한꺼번에 관리한 데다, A 씨가 1년 넘는 범행 기간 동안 경조사비 전달을 들쭉날쭉 임의로 했고 제대로 기록해두지 않아 누가 얼마만큼 돈을 전달받지 못한 것인지가 불분명하다는 것입니다.

외교부 관계자는 "받을 사람은 누가 얼마만큼 돈을 보냈는지를 모르고, 보낸 사람 역시 돈이 전달이 됐는지를 모르는 상황"이라며, 일일이 주인을 찾아주려 노력 중이지만 한계가 있다고 밝혔습니다.

이 관계자는 우선 '배달 사고'가 난 돈을 정리해 경조사비를 가장 많이 전달받지 못한 것으로 보이는 직원들에게 먼저 돈을 반환해오고 있다고 설명했습니다.

결국 A 씨가 항소심에서 감형을 위해 남은 돈 4천만 원 가량을 외교부에 모두 갚더라도, 그 돈은 이미 갈 곳을 잃은 만큼 '완전한' 변제가 되지 못할 가능성이 높아 보입니다.

    """
   test_review_list = ko_sentences_preproc([test_review_01, test_review_02, test_review_03])
   for test_review in test_review_list:
      prediction = predict_category(test_review, classifier, vectorizer, dataset._max_seq_length + 1)
      print("{} -> {}(p={:0.2f})".format(test_review, prediction['category'], prediction['probability']))
      print(vectorizer.category_vocab._token_to_idx.keys())