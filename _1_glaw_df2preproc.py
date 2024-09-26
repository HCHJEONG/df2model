# 1 word2vec 학습 자료
# 2 PREPROC FOR 이유 / 요지 임베딩 (for 문장 추천)

import re
import os
import kss
# import pickle5 as pickle
import pickle
import pandas as pd
from tqdm import tqdm

# from konlpy.tag import Kkma
# kkma = Kkma()
# import nltk
# nltk.download('punkt')
# from kiwipiepy import Kiwi
# kiwi = Kiwi()

IsReasoning = True
IsGists = True
min_charac_per_sentence = 30
min_charac_per_phrase = 20

# (있다|이다)(\([^\)]*\))?(\.)[ \n] 마침표는 3 group
delimiter_regex = '(었다|았다|였다|했다|있다|없다|한다|하다|이다|않다|아니다|같다|된다|됐다|옳다|쳤다|무겁다|가볍다|보다|본다|많다|있음|없음|함|임)(\([^\)]*\))?(\.)[ \n]'

# return [(listOfDictForSen, sentenceforwhat), (listOfDictForPhr, phraseforwhat)]
def split_sentence_phrase(df, field, num_workers, sen_phr_pair, phrase_also):

    if 'number' in df.columns:
        gistnoboolean = True
    else:
        gistnoboolean = False

    listOfDictForSen = []
    listOfDictForPhr = []
    # forwhat = field
    sentenceforwhat = "SentenceFor" + field
    phraseforwhat = "PhraseFor" + field

    for k in tqdm(range(len(df))):

        if type(df['case_full_no'].iloc[k]) == type('string') and len(df['case_full_no'].iloc[k]) != 0:
            pass
        else:
            continue

        cname = df['case_full_no'].iloc[k] + "_for_" + field
        # print("===============================")
        # print(f"case no: {cname}\n")

        ###########################################################################
        if str(type(df[field].iloc[k])) == "<class 'str'>" and len(df[field].iloc[k]) > 1:

            doc = df[field].iloc[k]

            # 1 판사 표시 부분 제거
            doc = remove_judges(doc)

            # 2 개행 문자를 복원
            doc = doc.replace("lnfd", "\n")

            # 3 (  )
            pattern = "\([\s]+\)"
            doc, _ = re.subn(pattern, '(그림)', doc)
            
            # 4 “   ”
            pattern = "“[\s]+”"
            doc, _ = re.subn(pattern, '“그림”', doc)

            # 5 [[별지1목록]어떤 내용1: 어떤 항목2]
            pattern = "\[\[[ 가-힣0-9]+\][ 가-힣0-9:]+\]"
            doc, _ = re.subn(pattern, ' ', doc)

            # 6 [박시환, 반대의견 : ]
            pattern = "\[[^\]]+\]"
            doc, _ = re.subn(pattern, ' ', doc)

            # 7 【이 유】
            pattern = "【[^】]+】"
            doc, _ = re.subn(pattern, ' ', doc)

            # 8 [4]
            pattern = "\[[0-9ⅰ-ⅻⅠ-Ⅻ]{1,2}\]"
            doc, _ = re.subn(pattern, ' ', doc)

            # 9 (4) 뒤 공백
            pattern = "\([0-9ⅰ-ⅻⅠ-Ⅻ]{1,2}\)\s"
            doc, _ = re.subn(pattern, ' ', doc)

            # 10 4) 앞 뒤 공백 
            pattern = "\s[0-9ⅰ-ⅻⅠ-Ⅻ]{1,2}\)\s"
            doc, _ = re.subn(pattern, ' ', doc)

            # 11 주4) 앞 뒤 공백
            pattern = "\s주[0-9ⅰ-ⅻⅠ-Ⅻ]{1,2}\)\s"
            doc, _ = re.subn(pattern, ' ', doc)

            # 12 4. 처음부분에 (앞에는 공백 없어도 되고 뒤에는 공백 있어야 하고)
            pattern = "([\s]*[0-9ⅰ-ⅻⅠ-Ⅻ]{1,2}\.\s)([^0-9.]{2,9})"
            if re.match(pattern, doc) == None:
                pass
            else:
                strt, nd = re.match(pattern, doc).span(1)
                doc_ = list(doc)
                for j in range(strt, nd):
                    doc_[j] = " "
                doc = ''.join(doc_)

            # 13 4. 중간에 (특수한 경우)
            pattern = "([가-힣)】\]%,“]\.{0,}\s{0,})([0-9ⅰ-ⅻⅠ-Ⅻ]{1,2}\.\s)"
            mtchlst = re.finditer(pattern, doc)
            doc_ = list(doc)
            for i in mtchlst:
                strt, nd = i.span(2)
                for j in range(strt, nd):
                    doc_[j] = " "
            doc = ''.join(doc_)

            # 14 4. 중간에
            pattern = "([^0-9.]{2,9})([0-9ⅰ-ⅻⅠ-Ⅻ]{1,2}\.\s)"
            mtchlst = re.finditer(pattern, doc)
            doc_ = list(doc)
            for i in mtchlst:
                strt, nd = i.span(2)
                for j in range(strt, nd):
                    doc_[j] = " "
            doc = ''.join(doc_)

            # 15 가) 앞 뒤 공백
            pattern = "\s[가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허a-zA-Z]\)\s"
            doc, _ = re.subn(pattern, ' ', doc)

            # 16 (가) 뒤 공백
            pattern = "\([가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허a-zA-Z]\)\s"
            doc, _ = re.subn(pattern, ' ', doc)

            # 17 가. 처음부분에 (앞에는 공백 없어도 되고 뒤에는 공백 있어야 하고)
            pattern = "([\s]+)?[가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허]\.\s"
            if re.match(pattern, doc) == None:
                pass
            else:
                strt, nd = re.match(pattern, doc).span()
                doc_ = list(doc)
                for j in range(strt, nd):
                    doc_[j] = " "
                doc = ''.join(doc_)

            # 18 나. 중간에
            pattern = "(\s{1,})([가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허]\.\s)"
            mtchlst = re.finditer(pattern, doc)
            doc_ = list(doc)
            for i in mtchlst:
                strt, nd = i.span(2)
                for j in range(strt, nd):
                    doc_[j] = " "
            doc = ''.join(doc_)
            
            # 19 특수문자 넘버링
            pattern = "[①-⓿㈀-㈛㉑-㊐○ⅰ-ⅻⅠ-Ⅻ]"
            doc, _ = re.subn(pattern, ' ', doc)

        else:
            # print("not str in field")
            continue
        ###########################################################################

        try:
            ss = kss.split_sentences(doc,
                                        backend="auto",
                                        num_workers=num_workers,
                                        )
            # ss = kiwi.split_into_sents(df['reasoning'].iloc[k])
            length_of_doc_sen = len(ss)
            # print(f"no of kss sentences: {length_of_doc_sen}\n")

        except Exception as e:
            # print(e)
            # print('kss error during sentence splitting...')
            # print(doc[:1000])
            # print()
            continue

        no = 0
        for idx, s in enumerate(ss): # ss는 kss로 나눈 문장들의 list

            ################################################################
            # nohangul = re.compile('[^ ㄱ-ㅣ가-힣.,0-9“”‘’]+') # 한글, 숫자, 인용부호, 콤마, 구두점과 띄어쓰기를 제외한 모든 글자
            # s = nohangul.sub(' ', s).strip()
            ################################################################

            if len(s.strip()) > min_charac_per_sentence:

                if gistnoboolean:
                    gistno = df['number'].iloc[k]
                else:
                    gistno = ""

                # 있다. 없다. 등으로 다시 구분해야 함
                regExStr = delimiter_regex
                regex_group_no = 3
                sub_sen_list = simple_ks_split_with_re(delimiter_regex, s, 3)
                # for ndx in list(range(1, len(delimiter_regex))):
                # sub_sen_list = onedim2twodim2onedim(sub_sen_list)

                # 문장의 길이 등 처리 후 dict에 저장
                for sss in [ x for x in sub_sen_list if  type(x) == type('string') and len(x) != 0]:

                    # 숫자가 너무 많은 경우 및 공백문자로만 된 경우 처리
                    if ( len(re.compile('\d').findall(sss)) > len(re.compile('[^\d]').findall(sss)) ) or len(sss.strip()) < 1:
                        # print("sentence too many digits or empty charac only not included!")
                        # print(sss)
                        # print()
                        continue                                
                    
                    # , 또는 . 으로 시작하는 문장은 첫 글자를 제거
                    if sss.strip()[0] == "," or sss.strip()[0] == ".":
                        sss = sss.strip()[1:]
                    sss = re.sub(",[ ,]*,", ' ', sss).strip()
                    
                    # 여러 공백 문자를 하나의 space 문자로 전환
                    sss = re.sub('[\s]+', " ", sss).strip()
                    
                    if len(sss) < min_charac_per_sentence:
                        # print("sentence too short not included!")
                        # print(sss)
                        # print()
                        continue
                    
                    ####################################################################
                    listOfDictForSen.append(
                        {"cname": cname, "idx": idx, "unit_str": sss, "gistno": gistno})
                    no = no + 1

                    # print("sentence long enough included!")
                    # print(sss)
                    # print()
                    ####################################################################

                    # 문장을 구절로 나누어 저장
                    if phrase_also:
                        # ss = doc.replace("lnfd", "\n")
                        # ss = ss.replace('. ', ', ') # 1999. 9. 9.의 경우 문제가 발생
                        regExStr = '[^0-9]([,])[^0-9]'
                        regex_group_no = 1
                        phr = simple_ks_split_with_re(regExStr, sss, regex_group_no)
                        
                        no_ = 0
                        str_buffer = ""
                        for idx_, p in enumerate(phr):

                            if p == None or (type(p) == type('p') and p == ''):
                                # print("phrase empty not included!")
                                # print(p)
                                # print()
                                continue

                            # 기본적으로 , 에서 나누지만 그 전후로 나뉜 것 각각을 strip()을 하고 내부에 공백이 없으면 나누면 안 됨
                            p = p.replace(",,", ',')
                            p = p.replace(",,", ',')
                            p = re.sub(",[ ,]*,", ' ', p).strip()

                            if type(p) == type('p') and len(p) == 0:
                                # print("phrase empty not included!")
                                # print(p)
                                # print()
                                continue

                            # , 또는 . 으로 시작하는 문장은 첫 글자를 제거
                            if p.strip()[0] == "," or p.strip()[0] == ".":
                                p = p.strip()[1:]

                            if type(p) == type('p') and len(p) == 0:
                                # print("phrase empty not included!")
                                # print(p)
                                # print()
                                continue
                            
                            # 짧은 길이 또는 괄호가 닫히지 않은 부분 기타 구절로 취급되기 어려운 경우들 버퍼에 담아 처리
                            if (re.compile('원고,$|피고,$|원고들,$|피고들,$|원고는,$|피고는,$|원고들은,$|피고들은,$|원고는 ,$|피고는 ,$|원고들은 ,$|피고들은 ,$').search(p.strip()) == None)\
                            and (re.compile('{[^}]+$').search(str_buffer + " " + p.strip())) == None\
                            and (re.compile('\([^)]+$').search(str_buffer + " " + p.strip())) == None\
                            and (re.compile('\[[^\]]+$').search(str_buffer + " " + p.strip())) == None\
                            and (' ' in p.strip())\
                            and (len(p.strip()) > min_charac_per_phrase):
                                
                                # print("phrase long enough included!")
                                # print(str_buffer + " " + p)
                                # print()
                                listOfDictForPhr.append(
                                        {
                                            "cname": cname, 
                                            "idx": idx_, 
                                            "unit_str": str_buffer + " " + p, 
                                            "gistno": gistno
                                        }
                                    )
                                no_ = no_ + 1
                                str_buffer = ""
                            else:
                                # print("phrase too short not included but stored in buffer!")
                                # print(p)
                                # print()
                                str_buffer = str_buffer + " " + p 
                        
                        if sen_phr_pair and no_ % 2 != 0:
                            listOfDictForPhr.pop()
                            # print("last odd phr removed!")

                    ####################################################################
            else:
                # print("sentence too short not included!")
                # print(s)
                # print()
                pass

        if sen_phr_pair and no % 2 != 0:
            listOfDictForSen.pop()
            print("last odd sen removed!")

    return [(postprocess_for_sen(listOfDictForSen), sentenceforwhat), (listOfDictForPhr, phraseforwhat)]
    
def remove_judges(doc):

    regex = re.compile('lnfd[ ]?(판사|군판사|대법관|대법원장|대법원판사|심판관|재판관|사법보좌관|재판장|대법원 강|대법원 조|제판장)[\[\](){}一-龥ㄱ-ㅣ가-힣 ]{1,150}(lnfd|\[|관계 법령|○)')
    doc = re.sub(regex, ' lnfd ', doc)
    return doc
    
def regExFindIter(regEx, text, i):
    tuplelist = []
    for xx in re.compile(regEx).finditer(text):
        matched = (lambda x: None if type(x) != re.Match else (x.group(i), x.span(i)))(xx)  # i 번째 그룹에 매치되는 문자열의 내용과 스트링 인덱스를 튜플로 반환하는 lambda 함수 / re.Match.span(0)은 역시 tuple을 반환함
        tuplelist.append(matched)
    return tuplelist

def simple_ks_split_with_re(regExStr, sen_not_splitted, regex_group_no):

    phr = []
    strt = 0
    residue_str = sen_not_splitted
    for xx in regExFindIter(regExStr, sen_not_splitted, regex_group_no):

        if xx !=None:
            content, idx = xx
            phr.append(sen_not_splitted[strt: idx[1]])
            residue_str = residue_str.replace(sen_not_splitted[strt:idx[1]], '')
            strt = idx[1]
    
    if len(residue_str) > 1:
        phr.append(residue_str)
    
    if len(phr) == 0:
        phr.append(sen_not_splitted)

    return phr

def postprocess_for_sen(listOfDics): 

    listOfDics_processed = []
    
    if len(listOfDics) == 0:
        print("len zero:")        
    else:

        str_buffer = ""
        unit_str = ""
        for _, u in enumerate(listOfDics):
            # u => {"cname": cname, "idx": idx, "unit_str": s, "gistno": gistno}

            if type(u['unit_str'])==type('unit_str') and len(u['unit_str']) > 1:

                if u['unit_str'].strip()[-1] == '.':                
                    unit_str = str_buffer + " " + u['unit_str']
                    str_buffer = ""

                    u_ = {"cname": u['cname'], "idx": u['idx'], "unit_str": unit_str, "gistno": u['gistno']}
                    listOfDics_processed.append(u_)
                    unit_str = ""
                else:
                    # 맨처음 후처리 시작할 때 . 으로 끝나지 않는 것은 그 다음 리스트 원소와 합해져야 함 예컨대 주소 요 가 등
                    str_buffer = str_buffer + " " + u['unit_str']
            else:
                continue    
 
    return listOfDics_processed

def save_pickle_xlsx(listOfDics_processed, forwhat, date_str):
    print('ten lines: ')
    print(listOfDics_processed[:10])
    ###########################################################################
    if not (os.path.isdir(f'..//web2df//dataset/{date_str}/')):
        os.makedirs(os.path.join(f'..//web2df//dataset/{date_str}/'))
    print('save as json file...')
    pd.DataFrame.from_records(listOfDics_processed).to_json(f'..//web2df//dataset//{date_str}//listForCase' + forwhat + '.json')
    print('save as pickle file...')
    with open(f'..//web2df//dataset//{date_str}//listForCase' + forwhat + '.pickle', 'wb') as f:
        pickle.dump(listOfDics_processed, f, pickle.HIGHEST_PROTOCOL)
    # pd.DataFrame.from_records(listOfDics_processed).to_excel('./dataset/listForCase' + forwhat + '.xlsx', index=False, header=True)
    ###########################################################################


if __name__ == "__main__":

    old_date_str = str(input("가장 최근에 진행했던 판례 업데이트 작업날짜를 적어주세요(예시:20240719 또는 adam):")) # 가장 최근에 수행한 업데이트 작업 날짜   
    date_str = str(input("판례 업데이트 작업을 진행하는 날짜를 적어주세요(예시:20240719):")) # 판례의 업데이트 작업을 시작한 날짜, 시작날짜는 모든 코드에서 통일되어야함.
    print(date_str)

    # preprocessing ##########
    # df_glaw_corpus = pd.read_json('..//web2df//saved//df_glaw_corpus_fullest.json')
    with open(f"..//web2df//saved//{date_str}//df_glaw_corpus//df_glaw_corpus_fullest_gmeta_lmeta.pickle", "rb") as fh:
        df_glaw_corpus = pickle.load(fh)
    print(df_glaw_corpus.info())

    # df_glaw_corpus = df_glaw_corpus.iloc[20000:20010]
    df_reasoning = pd.concat(
        [df_glaw_corpus['case_full_no'], df_glaw_corpus['reasoning']], axis=1)
    print(df_reasoning.info())  

    # df_glaw_summary = pd.read_json('..//web2df//saved//df_glaw_summary_fullest.json')
    with open(f"..//web2df//saved//{date_str}//df_glaw_corpus//df_glaw_summary_fullest.pickle", "rb") as fh:
        df_glaw_summary = pickle.load(fh)
    print(df_glaw_summary.info())

    # df_glaw_summary = df_glaw_summary.iloc[40000:40020]
    df_gists = pd.concat([df_glaw_summary['case_full_no'], df_glaw_summary['gists'], df_glaw_summary['number']], axis=1)
    print(df_gists.info())

    # input("data loaded... press enter for the next step...")

    if IsReasoning == True:

        for listOfUnits_processed, forwhat in split_sentence_phrase(df_reasoning, 'reasoning', 8, False, True):
            save_pickle_xlsx(listOfUnits_processed, forwhat, date_str)

    if IsGists == True:

        for listOfUnits_processed, forwhat in split_sentence_phrase(df_gists, 'gists', 8, False, True):
            save_pickle_xlsx(listOfUnits_processed, forwhat, date_str)

'''    
def simple_ks_split(sentences_str, delimiter):
    split = sentences_str.split(delimiter)
    return [substr + delimiter for substr in split[:-1]] + [split[-1]]

def onedim2twodim2onedim(sub_sen_list, delimiter_no):
    ssub_sen_list = []
    for idx, sen in enumerate(sub_sen_list):
        ssub_sen_list.append(simple_ks_split(sen, delimiters[delimiter_no]))
    return [data for inner_list in ssub_sen_list for data in inner_list]

delimiters = [
                '었다. ', 
                '았다. ',
                '였다. ', 
                '했다. ', 
                '있다. ', 

                '없다. ', 
                '한다. ', 
                '하다. ', 
                '이다. ', 
                '않다. ', 

                '아니다. ',
                '같다. ', 
                '된다. ', 
                '됐다. ',
                '옳다. ', 

                '쳤다. ', 
                '무겁다. ',
                '가볍다. ',
                '보다. ', 
                '본다. ', 

                '많다. ',
                '있음. ',
                '없음. ',
                '함. ',
                '임. ',

                '었다.\n', 
                '았다.\n',
                '였다.\n', 
                '했다.\n', 
                '있다.\n', 

                '없다.\n', 
                '한다.\n', 
                '하다.\n', 
                '이다.\n', 
                '않다.\n', 

                '아니다.\n',
                '같다.\n', 
                '된다.\n', 
                '됐다.\n',
                '옳다.\n', 

                '쳤다.\n', 
                '무겁다.\n',
                '가볍다.\n',
                '보다.\n', 
                '본다.\n', 

                '많다.\n',
                '있음.\n',
                '없음.\n',
                '함.\n',
                '임.\n',               
            ]

# 있다. 없다. 등으로 다시 구분해야 함
# sub_sen_list = simple_ks_split(s, delimiter_regex)
# for ndx in list(range(1, len(delimiter_regex))):
# sub_sen_list = onedim2twodim2onedim(sub_sen_list)

# def simple_ks_split(sentences_str, delimiter_regex):
#     # split = sentences_str.split(delimiter)
#     tuplelist = regExFindIter(delimiter_regex, sentences_str, 3)
#     split_sens = []
#     strt = 0
#     residue_str = sentences_str
#     for xx in tuplelist:
#         if xx !=None:
#             content, idx = xx
#             split_sens.append(sentences_str[strt: idx[1]])
#             residue_str = residue_str.replace(sentences_str[strt:idx[1]], '')
#             strt = idx[1]
                            
#     if len(residue_str) > 1:
#         split_sens.append(residue_str)
    
#     if len(split_sens) == 0:
#         split_sens.append(sentences_str)

#     return split_sens

# def onedim2twodim2onedim(sub_sen_list):
#     ssub_sen_list = []
#     for idx, sen in enumerate(sub_sen_list):
#         ssub_sen_list.append(simple_ks_split(sen, delimiter_regex))
#     return [data for inner_list in ssub_sen_list for data in inner_list]

'''